"""Archive extraction must never write outside its root or exhaust the disk."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from polymer_engine.core.errors import CorruptArchive, UnsafeArchiveMember
from polymer_engine.simulation.archive import detect_format, safe_extract


def tar_with(members: list[tuple[str, bytes]], path: Path, mode: str = "w:gz") -> Path:
    with tarfile.open(path, mode) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def zip_with(members: list[tuple[str, bytes]], path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members:
            zf.writestr(name, data)
    return path


class TestPathTraversal:
    def test_relative_traversal_is_blocked_tar(self, tmp_path: Path) -> None:
        archive = tar_with([("../escaped.txt", b"pwned")], tmp_path / "evil.tgz")
        with pytest.raises(UnsafeArchiveMember, match="escapes the extraction root"):
            safe_extract(archive, tmp_path / "out")
        assert not (tmp_path / "escaped.txt").exists()

    def test_deep_traversal_is_blocked(self, tmp_path: Path) -> None:
        archive = tar_with([("a/b/../../../../etc/passwd", b"x")], tmp_path / "evil.tgz")
        with pytest.raises(UnsafeArchiveMember):
            safe_extract(archive, tmp_path / "out")

    def test_absolute_path_member_is_blocked_tar(self, tmp_path: Path) -> None:
        archive = tar_with([("/tmp/absolute.txt", b"x")], tmp_path / "evil.tgz")
        with pytest.raises(UnsafeArchiveMember, match="absolute path"):
            safe_extract(archive, tmp_path / "out")

    def test_relative_traversal_is_blocked_zip(self, tmp_path: Path) -> None:
        archive = zip_with([("../escaped.txt", b"pwned")], tmp_path / "evil.zip")
        with pytest.raises(UnsafeArchiveMember):
            safe_extract(archive, tmp_path / "out")
        assert not (tmp_path / "escaped.txt").exists()

    def test_absolute_path_member_is_blocked_zip(self, tmp_path: Path) -> None:
        archive = zip_with([("/abs.txt", b"x")], tmp_path / "evil.zip")
        with pytest.raises(UnsafeArchiveMember):
            safe_extract(archive, tmp_path / "out")

    def test_windows_backslash_traversal_is_blocked(self, tmp_path: Path) -> None:
        archive = tar_with([("..\\..\\escaped.txt", b"x")], tmp_path / "evil.tgz")
        with pytest.raises(UnsafeArchiveMember):
            safe_extract(archive, tmp_path / "out")

    def test_nothing_is_written_when_one_member_is_unsafe(self, tmp_path: Path) -> None:
        """Validation covers the whole member list before any write happens."""
        archive = tar_with(
            [("good.txt", b"fine"), ("../evil.txt", b"bad")], tmp_path / "mixed.tgz"
        )
        destination = tmp_path / "out"
        with pytest.raises(UnsafeArchiveMember):
            safe_extract(archive, destination)
        assert not (destination / "good.txt").exists(), "a partial extraction leaks attacker content"


class TestLinks:
    def test_symlink_member_is_rejected(self, tmp_path: Path) -> None:
        archive = tmp_path / "link.tgz"
        with tarfile.open(archive, "w:gz") as tf:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
        with pytest.raises(UnsafeArchiveMember, match="link"):
            safe_extract(archive, tmp_path / "out")

    def test_hardlink_member_is_rejected(self, tmp_path: Path) -> None:
        archive = tmp_path / "hard.tgz"
        with tarfile.open(archive, "w:gz") as tf:
            payload = tarfile.TarInfo("real.txt")
            payload.size = 2
            tf.addfile(payload, io.BytesIO(b"hi"))
            info = tarfile.TarInfo("hard")
            info.type = tarfile.LNKTYPE
            info.linkname = "real.txt"
            tf.addfile(info)
        with pytest.raises(UnsafeArchiveMember):
            safe_extract(archive, tmp_path / "out")

    def test_broken_symlink_in_a_source_tree_is_not_followed(self, tmp_path: Path) -> None:
        from polymer_engine.simulation.system import discover_artifacts

        root = tmp_path / "sys"
        root.mkdir()
        (root / "real.gro").write_text("x\n1\n\n")
        (root / "dangling.itp").symlink_to(root / "missing.itp")
        artifacts = discover_artifacts(root)
        assert [a.relative_path for a in artifacts] == ["real.gro"]

    def test_zip_symlink_attribute_is_rejected(self, tmp_path: Path) -> None:
        archive = tmp_path / "link.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            info = zipfile.ZipInfo("link")
            info.external_attr = (0xA1FF << 16)  # S_IFLNK | 0777
            zf.writestr(info, "/etc/passwd")
        with pytest.raises(UnsafeArchiveMember, match="symlink"):
            safe_extract(archive, tmp_path / "out")


class TestSpecialFiles:
    @pytest.mark.parametrize("kind", [tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE])
    def test_device_members_are_rejected(self, tmp_path: Path, kind: bytes) -> None:
        archive = tmp_path / "dev.tgz"
        with tarfile.open(archive, "w:gz") as tf:
            info = tarfile.TarInfo("device")
            info.type = kind
            tf.addfile(info)
        with pytest.raises(UnsafeArchiveMember, match="device"):
            safe_extract(archive, tmp_path / "out")


class TestResourceLimits:
    def test_size_limit_is_enforced(self, tmp_path: Path) -> None:
        archive = tar_with([("big.dat", b"\0" * 50_000)], tmp_path / "big.tgz")
        with pytest.raises(UnsafeArchiveMember, match="size limit"):
            safe_extract(archive, tmp_path / "out", max_total_bytes=1000)

    def test_member_count_limit_is_enforced(self, tmp_path: Path) -> None:
        archive = tar_with([(f"f{i}.txt", b"x") for i in range(50)], tmp_path / "many.tgz")
        with pytest.raises(UnsafeArchiveMember, match="more members"):
            safe_extract(archive, tmp_path / "out", max_members=10)

    def test_a_compression_bomb_is_caught_before_writing(self, tmp_path: Path) -> None:
        """One highly compressible member must not be allowed to fill the disk."""
        archive = tar_with([("bomb.dat", b"\0" * 5_000_000)], tmp_path / "bomb.tgz")
        assert archive.stat().st_size < 100_000, "fixture should compress well"
        destination = tmp_path / "out"
        with pytest.raises(UnsafeArchiveMember):
            safe_extract(archive, destination, max_total_bytes=1_000_000)
        assert not (destination / "bomb.dat").exists()


class TestMalformedArchives:
    def test_missing_archive(self, tmp_path: Path) -> None:
        with pytest.raises(CorruptArchive, match="does not exist"):
            safe_extract(tmp_path / "nope.tgz", tmp_path / "out")

    def test_empty_file(self, tmp_path: Path) -> None:
        archive = tmp_path / "empty.tgz"
        archive.write_bytes(b"")
        with pytest.raises(CorruptArchive, match="empty"):
            safe_extract(archive, tmp_path / "out")

    def test_html_masquerading_as_an_archive(self, tmp_path: Path) -> None:
        archive = tmp_path / "session-expired.tgz"
        archive.write_text("<html><body>Please log in</body></html>")
        with pytest.raises(CorruptArchive):
            safe_extract(archive, tmp_path / "out")

    def test_truncated_gzip(self, tmp_path: Path) -> None:
        good = tar_with([("a.txt", b"x" * 5000)], tmp_path / "good.tgz")
        broken = tmp_path / "broken.tgz"
        broken.write_bytes(good.read_bytes()[: len(good.read_bytes()) // 2])
        with pytest.raises(CorruptArchive):
            safe_extract(broken, tmp_path / "out")

    def test_archive_with_no_members(self, tmp_path: Path) -> None:
        archive = tmp_path / "hollow.tgz"
        with tarfile.open(archive, "w:gz"):
            pass
        with pytest.raises(CorruptArchive, match="no members"):
            safe_extract(archive, tmp_path / "out")

    def test_unknown_format(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.rar"
        archive.write_bytes(b"Rar!\x1a\x07\x00 not really")
        with pytest.raises(CorruptArchive, match="Unrecognised archive format"):
            safe_extract(archive, tmp_path / "out")


class TestHappyPath:
    def test_valid_tar_extracts(self, tmp_path: Path) -> None:
        archive = tar_with([("sys/system.gro", b"title\n0\n\n"), ("sys/topol.top", b"[ molecules ]\n")], tmp_path / "ok.tgz")
        report = safe_extract(archive, tmp_path / "out")
        assert report.members_extracted == 2
        assert (tmp_path / "out" / "sys" / "system.gro").exists()
        assert report.archive_format == "tar"

    def test_valid_zip_extracts(self, tmp_path: Path) -> None:
        archive = zip_with([("a/b.txt", b"data")], tmp_path / "ok.zip")
        report = safe_extract(archive, tmp_path / "out")
        assert (tmp_path / "out" / "a" / "b.txt").read_bytes() == b"data"
        assert report.archive_format == "zip"

    def test_mislabelled_extension_is_sniffed(self, tmp_path: Path) -> None:
        archive = zip_with([("a.txt", b"x")], tmp_path / "actually-a-zip.bin")
        assert detect_format(archive) == "zip"
        assert safe_extract(archive, tmp_path / "out").members_extracted == 1

    @pytest.mark.parametrize("mode,suffix", [("w:", ".tar"), ("w:gz", ".tar.gz"), ("w:bz2", ".tar.bz2"), ("w:xz", ".tar.xz")])
    def test_all_tar_compressions(self, tmp_path: Path, mode: str, suffix: str) -> None:
        archive = tar_with([("a.txt", b"x")], tmp_path / f"a{suffix}", mode=mode)
        assert safe_extract(archive, tmp_path / f"out{suffix}").members_extracted == 1
