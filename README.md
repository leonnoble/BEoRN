# BEoRN: Bubbles during the Epoch Of Reionization Numerical-simulator

**BEoRN** is a simulation tool designed to model the state of the intergalactic medium (IGM) during cosmic dawn and reionization. It leverages semi-numerical methods to efficiently model the growth of ionized regions (bubbles) around early galaxies embedded in dark matter halos.

BEoRN is designed to be **flexible**, **user-friendly**, and **fast**, allowing researchers to efficiently explore various astrophysical scenarios and their impact on the 21-cm signal from neutral hydrogen.

**BEoRN is actively developed.** We welcome feedback and contributions. If you encounter any issues, please inform the developers by opening a [GitHub issue](https://github.com/cosmic-reionization/BEoRN/issues).

## Key Features

* **Lightweight & Modular:** A Python package suitable for simulation modules and analysis tools.
* **Data Visualization:** Utilities for assembling time/coeval cube data for easy visualization.
* **Reproducible:** Testing and CI-ready structure to support reproducible development.
* **Flexible Inputs:** Natively reads halo catalogs from simulations such as [Thesan](https://thesan-project.com/) and [PkdGrav](https://bitbucket.org/dpotter/pkdgrav3), or generates synthetic catalogs on the fly relying on [21cmFAST](https://github.com/21cmfast/21cmFAST).

## Documentation

Full documentation is available at: **[https://cosmic-reionization.github.io/BEoRN](https://cosmic-reionization.github.io/BEoRN)**

## Installation

### Standard Installation
You can install BEoRN directly using `pip`. This ensures all required dependencies are installed:
```bash
pip install git+[https://github.com/cosmic-reionization/beorn.git](https://github.com/cosmic-reionization/beorn.git)
```

### With 21cmFAST Support
To enable support for generating synthetic halo catalogs using the `21cmFAST` package, install with the extra option:
```bash
pip install "git+[https://github.com/cosmic-reionization/beorn.git](https://github.com/cosmic-reionization/beorn.git)[extra]"
```
**Note:** If this installation method fails, please refer to the [21cmFAST](https://github.com/21cmfast/21cmFAST) repository to install it properly, as it is being actively developed. Once `21cmFAST` is installed, you can proceed to install BEoRN using the standard method.

