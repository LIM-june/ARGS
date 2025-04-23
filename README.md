# Acoustic Rendering via Gaussian Splatting

Currently working on ARGS.

Based on Acoustic Volume Rendering (AVR) for Neural Impulse Response Fields
https://zitonglan.github.io/project/avr/avr.html

Visualizations based on MeshRIR codes
https://github.com/sh01k/MeshRIR

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Versions:
- Python : 3.10
- CUDA : 11.8

Installation:

```bash
git clone https://github.com/LIM-june/ARGS.git
cd ARGS
python -m venv .venv
. .venv/bin/activate
bash setup.sh
```

## Usage

Examples of how to use the project.

```bash
bash train.sh
bash infer.sh
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).