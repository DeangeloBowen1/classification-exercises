import env
import pandas as pd

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

def get_iris_data():
    return pd.read_sql('SELECT * FROM species', get_connection('iris_db'))

def get_telco_data():
    return pd.read_sql('''SELECT * FROM customers
                       join customer_payments using(customer_id)
                       join customer_contracts using(customer_id)
                       join customer_subscriptions using(customer_id)
                       ''',
                       get_connection('telco_churn'))
