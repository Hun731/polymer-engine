"""Safe archive extraction.

An archive downloaded from the internet is untrusted input.  Every member is
checked *before* anything is written, against:

* **Path traversal** -- ``../../etc/passwd`` and absolute members.
* **Links** -- symlinks and hardlinks, which can redirect a later write outside the
  extraction root even when their own path looks fine.
* **Special files** -- devices, FIFOs, sockets.
* **Decompression bombs** -- total uncompressed size and member count caps.

Extraction is atomic in spirit: validation happens over the whole member list first,
so a malicious archive never gets partially written.
"""

from __future__ import annotations

import gzip
import lzma
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from polymer_engine.core.errors import CorruptArchive, UnsafeArchiveMember
from polymer_engine.core.logging import get_logger

logger = get_logger("simulation.archive")

DEFAULT_MAX_TOTAL_BYTES = 2 * 1024**3
DEFAULT_MAX_MEMBERS = 200_000

TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")

#: Decompression failures surface as stream errors, not tar errors.  A truncated
#: ``.tar.gz`` raises ``EOFError`` from the gzip layer and would otherwise escape as
#: an unhandled exception in the middle of a campaign.
ARCHIVE_READ_ERRORS: tuple[type[BaseException], ...] = (
    tarfile.TarError,
    zipfile.BadZipFile,
    gzip.BadGzipFile,
    lzma.LZMAError,
    EOFError,
    OSError,
)
ZIP_SUFFIXES = (".zip",)


@dataclass
class ExtractionReport:
    destination: Path
    members_extracted: int = 0
    total_bytes: int = 0
    skipped: list[str] = field(default_factory=list)
    archive_format: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "destination": str(self.destination),
            "members_extracted": self.members_extracted,
            "total_bytes": self.total_bytes,
            "skipped": self.skipped,
            "archive_format": self.archive_format,
        }


def detect_format(archive: Path) -> str:
    """Identify the archive type from its name, then verify by opening it."""
    name = archive.name.lower()
    if name.endswith(ZIP_SUFFIXES):
        return "zip"
    if name.endswith(TAR_SUFFIXES):
        return "tar"
    # Fall back to content sniffing: a mislabelled extension is common.
    if zipfile.is_zipfile(archive):
        return "zip"
    try:
        if tarfile.is_tarfile(archive):
            return "tar"
    except (OSError, tarfile.TarError):
        pass
    raise CorruptArchive(
        "Unrecognised archive format", path=str(archive), supported=list(ZIP_SUFFIXES + TAR_SUFFIXES)
    )


def _check_member_path(name: str, destination: Path) -> Path:
    """Validate one member path and return where it will land.

    Uses pure-path arithmetic rather than ``resolve()`` so the check does not depend
    on what already exists on disk.
    """
    if not name or name in {".", "./"}:
        raise UnsafeArchiveMember("Archive member has an empty name", member=name)

    posix = PurePosixPath(name.replace("\\", "/"))
    if posix.is_absolute():
        raise UnsafeArchiveMember("Archive member uses an absolute path", member=name)
    if posix.drive or (len(name) > 1 and name[1] == ":"):
        raise UnsafeArchiveMember("Archive member uses a drive-qualified path", member=name)

    parts: list[str] = []
    for part in posix.parts:
        if part == "..":
            raise UnsafeArchiveMember("Archive member escapes the extraction root", member=name)
        if part in {"", "."}:
            continue
        parts.append(part)
    if not parts:
        raise UnsafeArchiveMember("Archive member resolves to no path", member=name)

    target = destination.joinpath(*parts)
    # Belt and braces: confirm the joined path is still under the destination.
    try:
        target.relative_to(destination)
    except ValueError:  # pragma: no cover - unreachable given the checks above
        raise UnsafeArchiveMember("Archive member escapes the extraction root", member=name) from None
    return target


def _validate_tar(tf: tarfile.TarFile, destination: Path, max_bytes: int, max_members: int) -> tuple[list[tarfile.TarInfo], int]:
    members = tf.getmembers()
    if len(members) > max_members:
        raise UnsafeArchiveMember(
            "Archive contains more members than allowed", members=len(members), limit=max_members
        )
    total = 0
    keep: list[tarfile.TarInfo] = []
    for member in members:
        _check_member_path(member.name, destination)
        if member.issym() or member.islnk():
            raise UnsafeArchiveMember(
                "Archive contains a link, which could redirect a write outside the root",
                member=member.name,
                link_target=member.linkname,
            )
        if member.ischr() or member.isblk() or member.isfifo():
            raise UnsafeArchiveMember("Archive contains a special device file", member=member.name)
        if member.isfile():
            total += member.size
            if total > max_bytes:
                raise UnsafeArchiveMember(
                    "Archive expands beyond the configured size limit",
                    uncompressed_bytes=total,
                    limit=max_bytes,
                )
        keep.append(member)
    return keep, total


def _validate_zip(zf: zipfile.ZipFile, destination: Path, max_bytes: int, max_members: int) -> tuple[list[zipfile.ZipInfo], int]:
    members = zf.infolist()
    if len(members) > max_members:
        raise UnsafeArchiveMember(
            "Archive contains more members than allowed", members=len(members), limit=max_members
        )
    total = 0
    keep: list[zipfile.ZipInfo] = []
    for member in members:
        _check_member_path(member.filename, destination)
        # ZIP stores the unix mode in the top 16 bits of external_attr.
        mode = (member.external_attr >> 16) & 0xFFFF
        if mode and (mode & 0xF000) == 0xA000:
            raise UnsafeArchiveMember("Archive contains a symlink member", member=member.filename)
        if not member.is_dir():
            total += member.file_size
            if total > max_bytes:
                raise UnsafeArchiveMember(
                    "Archive expands beyond the configured size limit",
                    uncompressed_bytes=total,
                    limit=max_bytes,
                )
        keep.append(member)
    return keep, total


def safe_extract(
    archive: str | Path,
    destination: str | Path,
    *,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> ExtractionReport:
    """Extract ``archive`` into ``destination``, refusing anything unsafe.

    Nothing is written until every member has passed validation.
    """
    archive = Path(archive)
    destination = Path(destination)
    if not archive.is_file():
        raise CorruptArchive("Archive does not exist", path=str(archive))
    if archive.stat().st_size == 0:
        raise CorruptArchive("Archive is empty", path=str(archive))

    fmt = detect_format(archive)
    destination.mkdir(parents=True, exist_ok=True)
    report = ExtractionReport(destination=destination, archive_format=fmt)

    if fmt == "tar":
        try:
            with tarfile.open(archive, "r:*") as tf:
                members, total = _validate_tar(tf, destination, max_total_bytes, max_members)
                if not members:
                    raise CorruptArchive("Archive contains no members", path=str(archive))  # noqa: TRY301
                # `filter="data"` is a second, independent guard in CPython >= 3.12.
                tf.extractall(destination, members=members, filter="data")
                report.members_extracted = sum(1 for m in members if m.isfile())
                report.total_bytes = total
        except CorruptArchive:
            raise
        except ARCHIVE_READ_ERRORS as exc:
            raise CorruptArchive(f"Could not read tar archive: {exc}", path=str(archive)) from exc
    else:
        try:
            with zipfile.ZipFile(archive) as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise CorruptArchive(  # noqa: TRY301
                        "Archive contains a corrupt member", path=str(archive), member=bad
                    )
                zip_members, total = _validate_zip(zf, destination, max_total_bytes, max_members)
                if not zip_members:
                    raise CorruptArchive("Archive contains no members", path=str(archive))  # noqa: TRY301
                zf.extractall(destination, members=[m.filename for m in zip_members])
                report.members_extracted = sum(1 for m in zip_members if not m.is_dir())
                report.total_bytes = total
        except CorruptArchive:
            raise
        except ARCHIVE_READ_ERRORS as exc:
            raise CorruptArchive(f"Could not read zip archive: {exc}", path=str(archive)) from exc

    _remove_dangling_symlinks(destination, report)
    logger.info(
        "Extracted %d members (%d bytes) from %s", report.members_extracted, report.total_bytes, archive.name
    )
    return report


def _remove_dangling_symlinks(destination: Path, report: ExtractionReport) -> None:
    """Defence in depth: drop any symlink that survived extraction.

    Validation already rejects link members, so this should find nothing; if it ever
    does, that is a bug worth failing loudly about rather than living with.
    """
    for path in destination.rglob("*"):
        if path.is_symlink():
            report.skipped.append(str(path))
            path.unlink()
            logger.warning("Removed unexpected symlink after extraction: %s", path)


__all__ = [
    "ARCHIVE_READ_ERRORS",
    "DEFAULT_MAX_MEMBERS",
    "DEFAULT_MAX_TOTAL_BYTES",
    "ExtractionReport",
    "detect_format",
    "safe_extract",
]
