# Dataset audit

- Metadata: `data/data_meta/metadata.csv`
- Rows: 40033 | classes: 2002 | views: {'front': 40033}
- Split sizes: {'train': 20016, 'test': 16014, 'val': 4003}
- Classes per split: {'test': 2002, 'train': 2002, 'val': 2002}
- Samples per class: min 13, max 20, mean 19.997; classes with < 20 samples: {'0/Zero': 13}
- Classes missing from a split: test: 0, train: 0, val: 0
- Invalid labels: 0
- Duplicate rows: 0 | duplicate video files: 0 | videos listed in more than one split: 0
- Missing video files: 0

## Signer independence

- Signer-independent splits: **True**
- test: 8 signers (s0004, s0006, s0010, s0014, s0016, s0017, s0019, s0020)
- train: 10 signers (s0001, s0002, s0005, s0007, s0008, s0009, s0011, s0012, s0013, s0018)
- val: 2 signers (s0003, s0015)
