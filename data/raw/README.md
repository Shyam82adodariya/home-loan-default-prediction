# data/raw

No data files are stored in this repository.

The notebook uses the public **Home Credit Default Risk** dataset from Kaggle: https://www.kaggle.com/c/home-credit-default-risk (a free Kaggle account and acceptance of the competition rules are needed to download it).

Place these seven files in this folder:

| File | One row represents |
|---|---|
| `application_train.csv` | one loan application (contains the `TARGET` column) |
| `bureau.csv` | one previous credit reported by another financial institution |
| `bureau_balance.csv` | one month of history for a previous bureau credit |
| `POS_CASH_balance.csv` | one month of history for a previous Home Credit loan |
| `credit_card_balance.csv` | one month of history for a previous credit card |
| `previous_application.csv` | one previous application to Home Credit |
| `installments_payments.csv` | one instalment payment record |

The notebook reads them from `../data/raw/`, relative to the `notebooks/` folder.
