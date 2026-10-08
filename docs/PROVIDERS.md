# Providers

Seven external data sources behind one interface. Adding a provider requires no change
to the orchestrator.

## Common contract

Every provider:

- declares its `capabilities` honestly — the registry reads them from the class, so it
  cannot advertise something the implementation does not do
- returns a `ProviderResult`, never raising for an expected failure
- distinguishes **"no records exist"** (`ok=True, records=[]`) from **"the call
  failed"** (`ok=False, error_type=...`)
- records request provenance: URL, status, timestamp, response SHA-256, attempt count,
  cache hit
- normalises records into a stable schema while keeping the raw payload for diagnosis

That third point matters more than it looks. If a rate limit or a DNS failure came
back as an empty list, a literature search would silently conclude that nothing has
been published on a topic.

### Error classification

| Condition | Exception | Retried? |
|---|---|---|
| HTTP 401 | `AuthenticationError` | no |
| HTTP 403 | `AuthorizationError` | no |
| HTTP 404 | `HttpStatusError` | no |
| HTTP 429 | `RateLimitError` | yes, honouring `Retry-After` |
| HTTP 5xx | `HttpStatusError` | yes |
| socket timeout | `TimeoutErrorProvider` | yes |
| connection refused / DNS | `TransportError` | yes |
| unparseable body | `ResponseFormatError` | no |
| capability not offered | `UnsupportedCapability` | no |

Retrying a 401 or a 404 only burns the provider's quota and hides the real problem, so
those never consume retry budget.

### Transport features

- exponential backoff with jitter, capped
- per-host token-bucket rate limiting
- content-addressed disk cache with TTL; error responses are never cached
- cache keys exclude `Authorization` headers, so a cache entry can neither be keyed by
  nor leak a credential
- offline mode: every uncached request raises rather than touching the network

---

## PubChem

*Contract:* <https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest> · no credentials · rate
limit 4 req/s (PubChem publishes a hard cap of 5)

**Capabilities:** `compound_lookup`, `compound_properties`

Resolves a compound by name, CID, SMILES or InChIKey and returns molecular formula,
weight, InChI(Key), SMILES and Lipinski counts.

Two behaviours worth knowing:

- A name with no match returns **HTTP 404 with a `Fault` body**. That is an empty
  result, not a failure, and is reported as `ok=True, records=[]`.
- PubChem renamed its SMILES fields (`CanonicalSMILES` → `SMILES`, `IsomericSMILES` →
  `ConnectivitySMILES`). Requesting a retired name yields HTTP 400. The client asks for
  the modern set first and falls back to the legacy set once, rather than pinning
  either and breaking.

---

## Crossref

*Contract:* <https://api.crossref.org> · no credentials · `CROSSREF_MAILTO` for the
polite pool

**Capabilities:** `literature_search`, `literature_metadata`

Bibliographic metadata by query or DOI. Records are deduplicated by DOI. A missing DOI
lookup is an empty result, not a failure.

> The base URL has **no `/v1` segment**. An earlier version of this repository used
> `https://api.crossref.org/v1`, which is not a documented path.

---

## Europe PMC

*Contract:* <https://www.ebi.ac.uk/europepmc/webservices/rest> · no credentials

**Capabilities:** `literature_search`, `fulltext_discovery`

Life-sciences literature with open-access status and full-text URLs. Full text is
*discovered*, not fetched. A well-formed zero-hit response may omit `resultList`
entirely; that is handled as empty rather than as a parse error.

---

## OpenAlex

*Contract:* <https://docs.openalex.org> · no credentials · `OPENALEX_MAILTO` for the
polite pool

**Capabilities:** `literature_search`, `literature_metadata`, `citation_graph`

Broad scholarly coverage. `referenced_works` is surfaced so the citation graph can be
walked without a second normalisation pass.

---

## RCSB PDB

*Contract:* search <https://search.rcsb.org/rcsbsearch/v2/query>, data
<https://data.rcsb.org/rest/v1/core>, files <https://files.rcsb.org/download> · no
credentials

**Capabilities:** `structure_search`, `structure_download`

> **Zero hits return HTTP 204 with an empty body**, not an empty JSON document.
> Treating that as a parse failure is a common bug; it is handled explicitly.

**Scope note.** The PDB holds experimentally determined *biomolecular* structures. For
synthetic polymers it is a source of reference conformations and comparison targets,
not of polymer entries.

---

## Materials Project

*Contract:* <https://api.materialsproject.org> · **requires `MP_API_KEY`**

**Capabilities:** `materials_reference`

**Scope note, because it matters scientifically.** Materials Project covers
**inorganic crystalline materials**. It is not a source of polymer data. Every record
is tagged `domain: "inorganic-crystalline"` so a downstream model cannot mistake an
inorganic entry for a polymer. Use it for fillers, substrates and composite
components.

The API key is held in a `Secret`, sent only as an `X-API-KEY` header, and never
appears in provenance — asserted by `tests/providers/test_data_providers.py`.

---

## CHARMM-GUI

*Contract:* <https://charmm-gui.org/?doc=api> · **requires credentials**

**Capabilities:** `job_login`, `job_status`, `job_download`
**Explicitly unsupported:** `job_submission`

| Method | Endpoint | Notes |
|---|---|---|
| POST | `/api/login` | email + password → JWT |
| GET | `/api/check_status?jobid=<ID>` | `pending` \| `running` \| `done` \| `error` |
| GET | `/api/download?jobid=<ID>` | `.tgz` archive |

All endpoints except login require `Authorization: Bearer <JWT>`. The token is valid
for up to 12 hours; the client refreshes 5 minutes early and treats a token whose
acquisition time is unknown as expired.

### Job submission is not available

CHARMM-GUI **publishes no job-submission endpoint**, and Polymer Builder is not
mentioned in its API documentation at all. `submit_module()` therefore raises
`UnsupportedCapability` and names the three endpoints that do exist. It does not guess
a path.

This is a real limitation of the upstream service, and the engine reports it as one.
The workflow is: build the system in the CHARMM-GUI web interface, then hand the
engine the job id.

```bash
polymer-engine provider charmm-gui status 1234567890
polymer-engine provider charmm-gui download 1234567890 ./job.tgz
polymer-engine system import ./job.tgz
```

### Safety behaviours

- An unrecognised status maps to `unknown` and is **polled again** — never optimistically
  read as completion.
- `wait_for_job` returns `ok=False, error_type="JobStillRunning"` on timeout rather
  than raising, so "still running" is a resumable outcome.
- Downloads are verified as readable tar archives before success is reported. An
  expired-session HTML page saved under a `.tgz` name is a classic way for a broken
  download to look like a valid scientific system; it is rejected as `CorruptArchive`.
- Job ids must be alphanumeric, which forecloses query injection.
- The JWT never appears in a result, in provenance, or in a log line.

---

## Testing

Every provider is tested against **recorded offline fixtures** in
`tests/fixtures/providers/`. The core suite never touches the network — an autouse
fixture makes real socket connections raise.

Each provider is exercised for the same matrix: valid response, empty result,
malformed result, HTTP 429, HTTP 5xx, timeout, connection failure, duplicate records,
cache hit and cache miss. CHARMM-GUI additionally covers successful login, invalid
credentials, expired token, 401/403/429/5xx, malformed JSON, all four job states,
download failure, corrupt archive and digest mismatch.

To add a fixture, record a real response, strip anything identifying, and save it as
JSON:

```python
transport = FixtureTransport()
transport.add_json("api.example.org", json.loads(fixture_path.read_text()))
provider = MyProvider(HttpClient(transport=transport))
```

## Adding a provider

```python
class InHouseProvider(Provider):
    name = "in_house"
    capabilities = frozenset({Capability.COMPOUND_LOOKUP})

    def health(self) -> ProviderResult:
        return self._guard("health", self._health)

    def lookup(self, identifier: str) -> ProviderResult:
        self.require(Capability.COMPOUND_LOOKUP)
        return self._guard("lookup", lambda: self._lookup(identifier))

registry.register("in_house", lambda config, http: InHouseProvider(http))
```

Declare only capabilities you have implemented. `_guard` converts known engine errors
into failed results; an *unexpected* exception propagates on purpose, because turning
a normalisation bug into "the provider returned nothing" is exactly the failure mode
that makes an autonomous pipeline draw wrong conclusions.
