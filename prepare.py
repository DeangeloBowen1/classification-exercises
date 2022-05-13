
"""
Contains CodeUp dataset functions for prepping data.
Author: Deangelo Bowen


Splt:

split_iris_data() :

splits iris data for species predition data

split_titanic_data() :

splits titanic data for survival prediction data

split_telco_data():

splits telco data for churn prediction data

Prep:

prep_iris_data():

preps and cleans iris data

prep_titanic_data():

preps and cleans titanic data

prep_telco_data():

preps and cleans telco data



"""




#split/prep iris data function------------------------------------------------------
def split_iris_data(iris):
    train_validate, test = train_test_split(iris, test_size=.2,
                                           random_state=123,
                                           stratify=iris.species)
    train,validate = train_test_split(train_validate, test_size=.3,
                                     random_state=123,
                                     stratify=train_validate.species)
    return train, validate, test

def prep_iris(iris):
    iris = iris.drop(['species_id','measure_id'], axis = 1)
    iris = iris.rename(columns={'species_name': 'species'})
    dummy_var = pd.get_dummies(iris[['species']], dummy_na = False, drop_first=True)
    iris = pd.concat([iris, dummy_var], axis = 1)
    return


#---------------------------------------------------------------------------------






#split/prep titanic data function------------------------------------------------------
def split_titanic_data(titanic):
    train_validate, test = train_test_split(titanic, test_size=.2,
                                           random_state=123,
                                           stratify=titanic.survived)
    train,validate = train_test_split(train_validate, test_size=.3,
                                     random_state=123,
                                     stratify=train_validate.survived)
    return train, validate, test

# prep titanic data
def prep_titanic(titanic):
    titanic = titanic.drop(['passenger_id',
                            'class','deck', 'embarked'], axis= 1)
    titanic.drop_duplicates(inplace=True)
    titanic['age'] = titanic.age.fillna(titanic.age.mean())
    titanic['embark_town'] = titanic.embark_town.fillna('Southampton')
    dummy_titanic = pd.get_dummies(titanic[['sex', 'embark_town']],
                              dummy_na = False,
                              drop_first = [True, True])
    titanic = titanic.drop(['sex','embark_town'], axis= 1)
    titanic = pd.concat([titanic, dummy_titanic], axis = 1)
    return titanic
#------------------------------------------------------------------------------------





#split/prep telco data function--------------------------------------------------------
def split_telco_data(telco):
    train_validate, test = train_test_split(telco, test_size=.2,
                                           random_state=123,
                                           stratify=telco.churn)
    train,validate = train_test_split(train_validate, test_size=.3,
                                     random_state=123,
                                     stratify=train_validate.churn)
    return train, validate, test





#Prep telco data function
def prep_telco(telco):
    
    # drop duplicates
    telco.drop_duplicates(inplace=True)
    
    # drop specific columns
    telco = telco.drop(['internet_service_type_id', 'payment_type_id', 'contract_type_id',
                  'customer_id', 'online_security', 'online_backup', 'device_protection',
                  'tech_support', 'streaming_tv', 'streaming_movies'], axis= 1)
    
    # strip spaces from total charges, turn into a float
    telco['total_charges'] = telco['total_charges'].str.strip()
    telco = telco[telco.total_charges != '']
    telco['total_charges']= telco.total_charges.astype('float')
    
    # map Yes = 1, No = 0
    telco['partner'] = telco.partner.map({'Yes': 1, 'No': 0})
    telco['dependents'] = telco.dependents.map({'Yes': 1, 'No': 0})
    telco['phone_service'] = telco.phone_service.map({'Yes': 1, 'No': 0})
    telco['paperless_billing'] = telco.paperless_billing.map({'Yes': 1, 'No': 0})
    telco['churn'] = telco.churn.map({'Yes': 1, 'No': 0})
    
    # create dummy for cat cols to numeric cols
    dummy_telco = pd.get_dummies(telco[['gender', 'multiple_lines', 'contract_type',
                                     'payment_type', 'internet_service_type']])
    
    # concat
    telco = pd.concat([telco, dummy_telco], axis = 1)
    
    # train validate test
    train, validate, test = split_telco_data(telco)
    
    return train, validate, test
#---------------------------------------------------------------------------------
