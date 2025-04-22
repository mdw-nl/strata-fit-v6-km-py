<h1 align="center">
  <br>
  <a href="https://vantage6.ai"><img src="https://github.com/IKNL/guidelines/blob/master/resources/logos/vantage6.png?raw=true" alt="vantage6" width="400"></a>
</h1>

<h3 align=center> A privacy preserving federated learning solution</h3>

--------------------


# Kaplan-Meier implementation for STRATA-FIT
Computes Kaplan-Meier curve for STRATA-FIT specific datasets via [vantage6](https://vantage6.ai). To learn more about STRATA-FIT please visit [STRATA-FIT.eu](https://strata-fit.eu/).


## Running the Federated KM Mock‑Client Test

### 1. Install Git and Python  
- **Git**: download and install from https://git‑scm.com/downloads  
- **Python 3.10+**: install from https://python.org/downloads  
- Verify installation by opening a terminal (macOS/Linux) or PowerShell (Windows) and running:  
  ```bash
  python --version  
  ```

### 2. Clone the Repository  
- Navigate to the folder where you want the code  
- Run:
  ```
  git clone https://github.com/mdw-nl/strata-fit-v6-km-py  
  ```
- Change into the project directory:  
  ```bash
  cd strata-fit-v6-km-py
  ```

### 3. Create & Activate a Virtual Environment  
- **macOS/Linux**:  
  ```bash
  python3 -m venv .venv  
  source .venv/bin/activate
  ```  
- **Windows (PowerShell)**:
  ```powershell  
  python -m venv .venv  
  .venv\Scripts\Activate.ps1
  ```

### 4. Install Dependencies  
- Upgrade pip:
  ```bash
  pip install --upgrade pip
  ```
- Install requirements:  
  ```bash
  pip install -r requirements.txt
  ```

### 5. Prepare Your Three Node Datasets  
- Place your three STRATA‑FIT CSV exports (or the provided `alpha.csv`, `beta.csv`, `gamma.csv`) into `tests/data/data_times/` 
- The mock client will treat each file as one “organization.” 

### 6. Execute the Mock Client test
- Run:
  ```bash
  python tests/mock_client.py
  ```

You should see INFO logs, the first five rows of the KM table, summary statistics, and a step plot of cumulative incidence of D2T‑RA over years since diagnosis.

---

### Splitting Your Own Data  
If you have one large CSV, split it into three files (e.g. equal row chunks or by patient ID) named alpha.csv, beta.csv, gamma.csv in tests/data/data_times/. The mock client will treat them as three separate nodes.  

This works on any OS. Simply follow these steps to validate your federated KM implementation end‑to‑end.  

------------------------------------
> [vantage6](https://vantage6.ai)