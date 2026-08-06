# Embedding Code Transfer Inventory

**Status:** `TRANSFER_BUNDLE_READY` / `IMPLEMENTED_NOT_EXECUTED` / `TRANSFER_PENDING_VALIDATION`

Generated (UTC): `2026-08-04T13:23:07Z`

## A. Required code files

| Repository-relative path | Size (bytes) | SHA-256 | Required | Purpose |
|---|---:|---|---|---|
| `configs/embeddings/mock_end_to_end.yaml` | 1154 | `07233d92577770be340834b6f5f5e934e1d54023a9e561e1cc94a379f1c74291` | required | transfer member |
| `configs/embeddings/qwen3_bounded_pilot_10k.yaml` | 1644 | `6daaffaf0422a8be953704701bae2ffcc366545e643331afecbe312174cc9096` | required | transfer member |
| `configs/embeddings/qwen3_preflight_64.yaml` | 2077 | `1152e67fdf6ac2381e0c6d18658aae1713a5c78fca22e5b9bf7147d0cff0832e` | required | transfer member |
| `docs/handoff/10_target_studio_embedding_runbook.md` | 9636 | `0a5281837f43076f0d226d37275b3d3d710351a88ca2453d74c6cbdcd4fecb1a` | required | transfer member |
| `docs/implementation/embedding_stage_implementation_notes.md` | 2109 | `947c7bd5711a1417ab2b24d7964de26e0b4e5b610389d06490697e13be1755d5` | required | transfer member |
| `docs/method/16_q_emb_embedding_contract_and_pilot_spec.md` | 19804 | `de345583f2a91e335cd736f4268b399743ba90b692fabd830bbb8799a112c35c` | required | transfer member |
| `pyproject.toml` | 1691 | `6e62fe6c2afe6140e2de163d88754ddec851c3c7042eec1ca53a5aa329b08ac5` | required | transfer member |
| `requirements/embeddings-target-studio.txt` | 1782 | `820a9e9217de5ec2508aa1693ebb06ed2cf08d0599c2c2cf37880ea7943fab28` | required | transfer member |
| `src/tdmec/__init__.py` | 608 | `ef209bbd7d2e0bd545b99153d6c93bf97bda404f906bb2b69407e48a5fd3354c` | required | transfer member |
| `src/tdmec/config/__init__.py` | 804 | `a4c683def51a7c38adeeb86b506841ac957c30e1b1aca5565b7be579fd35daf7` | required | transfer member |
| `src/tdmec/constants.py` | 5750 | `b0506af92189a0b36a59492121b9a265258446357cf7f7c7b42ab1299c4a015a` | required | transfer member |
| `src/tdmec/fixtures/__init__.py` | 107 | `93be5420dcc672a08e2f3209a40d2e6d098aa05dfd5cddc40cfa1c83090c644c` | required | transfer member |
| `src/tdmec/hashing.py` | 8176 | `4b07c072bc7ab1b832151891b4edd7b6346dc559c06af6b00765fdc8d0691771` | required | transfer member |
| `src/tdmec/schemas/__init__.py` | 760 | `325f7ad6bc6e6b21f9031c2865ab8b0dd78d8f573762ef00d8315fa63ae23547` | required | transfer member |
| `src/tdmec/unresolved.py` | 1525 | `8fc36f0f54c2d7381026da31a0e90b4898ab0e2f8f1cb43262478465cca58304` | required | transfer member |
| `src/tdmec/validation/__init__.py` | 335 | `7456fe8bf84e8b9de9c2346941b93607c6935d4ebd404d7c437b04cabebf6728` | required | transfer member |
| `src/tdmec_embeddings/__init__.py` | 308 | `690219587d66df57b5b05c98815a3a231cef8da00bd3c04b304118572ab03cb3` | required | transfer member |
| `src/tdmec_embeddings/cli.py` | 2578 | `87a26bd2e4f0261dfa53a336ede1e96e2b6589a4b45ffb8c7b96f353f5db19cd` | required | transfer member |
| `src/tdmec_embeddings/config.py` | 10217 | `32ab329cf2504766a814e5bb35d7a462cbe4eaa96cfb24eccb5ed0a422bb4f79` | required | transfer member |
| `src/tdmec_embeddings/eligibility.py` | 21399 | `481a2658eae402156f8167942c3edfe907d58fd301b1f1e4758484040413c04a` | required | transfer member |
| `src/tdmec_embeddings/file_cli.py` | 3875 | `13ce74cd865fd58b8570e3e775d3ad8551b1af14f059b43cc99128ebd11aab5d` | required | transfer member |
| `src/tdmec_embeddings/file_sources.py` | 21892 | `eb695fff06dfd30537d579c0eab68c1dd690252d55a3784645e10fcc8a71e787` | required | transfer member |
| `src/tdmec_embeddings/file_writer.py` | 30053 | `15df387d0e0785dfc087b1e12d18c6ca1504129470c0a393cdb44f0b36b143c4` | required | transfer member |
| `src/tdmec_embeddings/implementation_status.py` | 348 | `27cbab5d573c5f6de3cbd93b6bd1ebc5516384d92864096e7e38854ff3026ebb` | required | transfer member |
| `src/tdmec_embeddings/mock_encoder.py` | 5284 | `444d941c890f70bb2f16da2bf18331b93091eb1646ba9e5cf3759b4a2cf68373` | required | transfer member |
| `src/tdmec_embeddings/pipeline.py` | 13847 | `8f274f24bac11015973c7146e471f39472bdced966bd1b7e76b568f1a68ca8f6` | required | transfer member |
| `src/tdmec_embeddings/pooling.py` | 25525 | `02d7ce132faaaee86b17d83cd8ad94d28c00d04e07807b5e6905d0d8f0a44d09` | required | transfer member |
| `src/tdmec_embeddings/qwen_encoder.py` | 14184 | `342c46c634c24b20f96141bb51df0cb121ccc93671a2bf8783e5864a2e75f09e` | required | transfer member |
| `src/tdmec_embeddings/reader.py` | 1091 | `2f1d8b1b4c76943b8fa12ae6cce3a95dd28def014da8691c15817ca2ab3f2ada` | required | transfer member |
| `src/tdmec_embeddings/sampling.py` | 6909 | `df414c39930fbd3818303726996f39ae116ea9ba1aa8be757f604b0f63da06ec` | required | transfer member |
| `src/tdmec_embeddings/writer.py` | 1730 | `c7068aa0c87e29635cfe310e5f0ed7b0db5f06ce30695785f8270f85f5975e08` | required | transfer member |
| `tests/test_embeddings_eligibility.py` | 10504 | `6ffa95d9e21b6ea2e414725756ee33c7c3beb12cacf50aabdd8256e72264725e` | required | transfer member |
| `tests/test_embeddings_file_sources.py` | 9061 | `0e76a622d6dcbf223f9f79c55025afcd7770fa87e0bc9973f3aa923f2b5ec144` | required | transfer member |
| `tests/test_embeddings_file_writer.py` | 11491 | `f792e23b1ee17f5a58c84be3eb56767e6c0741af5ee569be7d466e02eb43b3ff` | required | transfer member |
| `tests/test_embeddings_pipeline.py` | 4249 | `fe3947953c8a8aa6a65c149c9372024bba21d85387dcd461702f2ffff502d6e7` | required | transfer member |
| `tests/test_embeddings_pooling.py` | 8054 | `6f8ac8f09c5e76593f49a64d79d10574055c43484fab17579aaf679fd48d4df2` | required | transfer member |
| `tests/test_embeddings_qwen_encoder.py` | 7242 | `568ebc02330c849276eb2476ae6923111e015d5c0558576e74b61f8980701618` | required | transfer member |
| `tests/test_embeddings_sampling.py` | 4013 | `d1cfaf8eb32d8b8993706a1a5fcdfd57ec04c1fc3e8b0e503d7441420eb1540e` | required | transfer member |
| `transfer/EXTRACT_AND_INSTALL.md` | 1840 | `97f1b9d880c6fe858aae3dc3af20f00b7b3188f1d2e65e4e8d836ea670563313` | required | transfer member |
| `transfer/TRANSFER_INVENTORY.md` | 7314 | `a900f115a493d686b2110fa16f2f729f7c4f40984562adb573a1fc40491fb583` | required | transfer member |
| `transfer/TRANSFER_MANIFEST.json` | 14430 | `1b63739b5af00183f42f9e3a5ae995d15a624910ed8f4814c9f00dabac36587a` | required | transfer member |
| `transfer/pyproject.embeddings-only.toml` | 1116 | `6feb27a3f4804b0a8b1975e6640e2afec0078c21212c2f2591f3b45c2386ab7e` | required | transfer member |
