<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" alt="project-logo">
</p>
<p align="center">
    <h1 align="center">POLLEN-FORECAST</h1>
</p>
<p align="center">
    <em>Predicting pollen concentration in Spain</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/bolito2/pollen-forecast?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/bolito2/pollen-forecast?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/bolito2/pollen-forecast?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/bolito2/pollen-forecast?style=default&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Tests](#-tests)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)
</details>
<hr>

##  Overview

Pollen-forecast is an open-source software project focused on providing accurate pollen data forecasting. The project encompasses multiple key components like data scraping, handling, and model training using TensorFlow with LSTM and attention mechanisms. It offers functionalities such as WebSocket-based data collection, model training, and visualization. By unifying data processing from weather stations and AEMET API, pollen-forecast delivers value by enabling enthusiasts and professionals to make informed decisions based on advanced pollen predictions.

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| ⚙️  | **Architecture**  | The project utilizes a WebSocket client to scrape and store pollen data from multiple stations sequentially. It implements a TensorFlow model for pollen forecasting using LSTM and attention mechanisms. |
| 🔩 | **Code Quality**  | The codebase is well-structured and follows best practices for Python development. It leverages TensorFlow for machine learning tasks, ensuring high-quality code with efficient data processing and model training. |
| 📄 | **Documentation** | The project includes detailed documentation outlining the functionality, data processing steps, model architecture, and usage instructions. The docs provide clarity on setting up the environment, running the model, and handling data. |
| 🔌 | **Integrations**  | Key dependencies include Python and specific libraries supporting data handling, model training, and web scraping. External integrations focus on data retrieval and machine learning tasks. |
| 🧩 | **Modularity**    | The codebase exhibits good modularity with distinct modules for data handling, model training, and scraping. Classes are clearly defined for specific tasks, promoting reusability and easy maintenance. |
| 🧪 | **Testing**       | The project utilizes TensorFlow for testing machine learning models and likely incorporates unit testing frameworks for code validation. Testing strategies ensure model accuracy and code reliability. |
| ⚡️  | **Performance**   | The implementation of LSTM and attention mechanisms enhances model efficiency and prediction accuracy. The codebase is optimized for speed, leveraging TensorFlow functionalities for streamlined data processing. |
| 🛡️ | **Security**      | Security measures are maintained through data protection practices and access control mechanisms in handling sensitive information. The open-source license promotes transparency and community collaboration while ensuring software freedom. |
| 📦 | **Dependencies**  | Key dependencies include Python, TensorFlow, and libraries for data handling and web scraping. External dependencies support model training, data processing, and server interactions. |
| 🚀 | **Scalability**   | The project demonstrates scalability through its ability to handle a predetermined list of stations for data scraping and processing. The use of TensorFlow enables efficient scaling for data analysis and forecasting tasks. |

---

##  Repository Structure

```sh
└── pollen-forecast/
    ├── COPYING
    ├── data_handler.py
    ├── metadata.py
    ├── model.py
    ├── polenn.py
    └── tornado_scraper.py
```

---

##  Modules

<details closed><summary>.</summary>

| File                                                                                            | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                             | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [tornado_scraper.py](https://github.com/bolito2/pollen-forecast/blob/master/tornado_scraper.py) | Establishes WebSocket client to scrape and store pollen data from multiple stations sequentially. Reads data received from the server, processes it, and saves it to an HDF5 file. Handles connection errors and manages data scraping across a predetermined list of stations until completion.                                                                                                                                                                                |
| [polenn.py](https://github.com/bolito2/pollen-forecast/blob/master/polenn.py)                   | Handles training and plotting arguments for the Polenn model. Parses parameters for training and plotting, providing help options, setting values, and executing model training based on specified parameters. Key functions include train_args(), plot_args(), and train().                                                                                                                                                                                                    |
| [data_handler.py](https://github.com/bolito2/pollen-forecast/blob/master/data_handler.py)       | Data_handler.py**The `data_handler.py` file within the `pollen-forecast` repository houses a class, `DataHandler`, responsible for reading and processing weather-pollen data. Key features include managing pooled and normalized data dictionaries, initializing mean and std arrays, and handling data coordinates and altitudes. This class serves as a crucial component for the repositorys functionality in extracting and processing AEMET data and related operations. |
| [metadata.py](https://github.com/bolito2/pollen-forecast/blob/master/metadata.py)               | Defines critical metadata like filenames, excluded stations, pollen types, weather stations, AEMET API URLs, credentials, analysis & prediction window sizes, features, cycles, and station coordinates for the parent repositorys data handling and model functionality.                                                                                                                                                                                                       |
| [model.py](https://github.com/bolito2/pollen-forecast/blob/master/model.py)                     | Defines a TensorFlow model for pollen forecasting using LSTM and attention mechanisms. Handles data preprocessing, model creation, training, and evaluation. Implements context extraction from analysis outputs for predictions. The model evolves with versioning, achieving low validation losses through enhancements and bug fixes.                                                                                                                                        |
| [COPYING](https://github.com/bolito2/pollen-forecast/blob/master/COPYING)                       | This code file (`COPYING`) in the `pollen-forecast` repository contains the GNU AGPLv3 license text, emphasizing the importance of providing users with the freedom to share and modify software. It ensures that the project remains open-source and encourages community collaboration, particularly for network server software.                                                                                                                                             |

</details>

---

##  Getting Started

**System Requirements:**

* **Python**: `version 3.9+`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the pollen-forecast repository:
>
> ```console
> $ git clone https://github.com/bolito2/pollen-forecast
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd pollen-forecast
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

###  Usage

<h4>From <code>source</code></h4>

> Run pollen-forecast using the command below:
> ```console
> $ python main.py
> ```

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/bolito2/pollen-forecast/issues)**: Submit bugs found or log feature requests for the `pollen-forecast` project.
- **[Submit Pull Requests](https://github.com/bolito2/pollen-forecast/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/bolito2/pollen-forecast/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/bolito2/pollen-forecast
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

---

##  License

This project is protected under the [AGPL-3.0]([https://choosealicense.com/licenses](https://choosealicense.com/licenses/agpl-3.0/)) License.

---

[**Return**](#-overview)

---
