# Adaptive Enhanced Online Scheduling Algorithm

This project implements an **Adaptive Enhanced Online Algorithm** for scheduling big data jobs in a cloud environment. It improves traditional online scheduling methods by adapting to job characteristics and penalty conditions.

---

## 🧩 Components

- `job.csv`: Raw job trace file (input)
- `Excel_process.py`: Preprocesses `job.csv` and outputs `output.csv`
- `output.csv`: Cleaned and formatted job data for scheduling
- `E11.py`: Implements the Adaptive Enhanced Online Scheduling Algorithm

---

## ⚙️ Requirements

- Python 3.x
- Required libraries:
  - `pandas`
  - `numpy`

Install using:

````bash
pip install pandas numpy```

🚀 How to Run
Step 1: Preprocess job data

```bash
python Excel_process.py```

Reads job.csv

Generates output.csv with cleaned job information

Step 2: Run the Adaptive Scheduler

```bash
python E11.py```

Reads output.csv

Applies the adaptive enhanced scheduling logic

Displays the output graphs
````
