# FAIR release preparation

Two linked Zenodo records are planned: **models** (encoders, decoder, PCA and
calibration) and **context** (training pairs, an example query and the energy
axis). Both retain restricted file access. Kumo's upstream weights stay at
their pinned upstream URL. No raw images or held-out test rows are packaged.

Creator: **Daniel Vega Bermudez**, ORCID
[0009-0002-0548-0800](https://orcid.org/0009-0002-0548-0800), verified against the
ORCID public API on 2026-10-15. Access contact: **daniel@inversionsemi.com**.

## Metadata and scope

`metadata.json` is the editable release plan. The package builder produces:

* `manifest.json`: exact file inventory, SHA-256, sizes, source hashes, provenance,
  proposed license, access conditions, and reviewed code commit.
* `schema.json`: array shapes/dtypes, feature/target order, scalar definitions,
  known units, and explicit unit-verification gaps.
* `dataset.jsonld`: Schema.org Dataset JSON-LD with creator ORCID, description,
  version, distributions, code provenance and contact. Reserved version DOIs
  and reciprocal record links are included when available.
* `checksums.json`: SHA-256 for every artifact and accompanying metadata file
  other than the checksum manifest itself.
* `publication-status.json`: explicit unresolved publication requirements.

Framework-native `.pt` files are PyTorch `state_dict` dictionaries and tensor
normalization, loadable with `torch.load(..., weights_only=True)`, without custom
pickled model classes. NPZ/NPY files are NumPy numeric arrays, loadable with
`allow_pickle=False`. PCA files retain the fixed mean, components, crop, shape,
fit-row indices, and variance information where present. See the source
package README for image preparation, interfaces, dependencies, and inference.

The context contains 2,152 aligned training rows: `train_x` (2152×88), `train_y`
(2152×21), and `train_indices`. It preserves the original feature values;
consumers must use the matching encoder versions. The prepared example is from
validation and is separate from the context. There are no test-set targets or
raw images in either release.

Model limitations: this is a frozen diagnostic for its measured operating
regime. The current Kumo server fixes context size and feature/target widths.
Four ensemble members are not a calibrated uncertainty interval. CPU encoder
outputs may differ slightly between numerical backends; artifact extraction
was checked against the original implementations. Validation comparisons do
not establish accuracy outside the source regime.

## Licensing provenance

The earlier Genesis DMSP draft, sections 3.2–3.4, proposes Apache-2.0 for public
model packages/software and CC0-1.0 or CC-BY-4.0 for datasets outside Dryad.
The July 16 revised plan requires appropriate licenses selected per release,
without fixing a single SPDX identifier. Accordingly, these drafts propose:

| Record | Proposed SPDX license |
| --- | --- |
| Encoder/decoder artifacts | Apache-2.0 |
| Context data | CC-BY-4.0 |

These are proposed release terms, not a claim of completed release clearance.
They do not change the existing code repository's proprietary license notice.
Confirm the source-data/model sharing authority and final terms before deposit;
then include the full applicable license text and update `license_status`.
Upstream Kumo/SDM retains its own license and attribution.

## FAIR publication requirements

This preparation follows [Zenodo's FAIR principles](https://about.zenodo.org/principles/).
The local drafts are **not yet published FAIR records**.

| Principle | Prepared | Still required before publication |
| --- | --- | --- |
| Findable | Descriptions, versions, creator ORCID, keywords and file inventory | Reserve specific-version DOIs, add them to both records and manifests, publish/index metadata |
| Accessible | Restricted-file intent, named contact, explicit request procedure, no embedded credentials | Configure and verify actual Zenodo access and metadata retrieval |
| Interoperable | JSON/JSON-LD, array schemas, original column order, SPDX identifiers and source references | Verify instrument-native scalar/energy units; add reciprocal DOI links |
| Reusable | Checksums, provenance, source hashes, code revision, preprocessing, limitations and proposed licenses | Confirm release authority/terms, include license texts and complete the model/data record review |

Authentication for restricted files is compatible with FAIR; public metadata
must remain available. Do not claim file access or a DOI exists before it does.
Do not use the concept DOI or a moving `latest` link to resolve runtime weights:
pin the exact version, then verify the SHA-256 from Git.

## Prepare locally

After committing the diagnostic code, from the repository root:

```bash
python lpa/Virtual_Diagnostics/release/prepare.py \
  --artifacts local/virtual_diagnostics/artifacts \
  --output local/virtual_diagnostics/zenodo-draft-v1
```

The builder checks every artifact, refuses overwrite, separates model/context
files, and records the exact committed code revision. It has no upload or
publish operation. Keep the prepared directories private until the record's
metadata, rights, units, DOI links and access configuration are complete.
After an authorized deposit, verify the downloaded files and metadata, then
update the checked-in runtime manifest with the actual specific-version DOIs.
