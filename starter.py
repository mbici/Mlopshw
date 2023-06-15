

import pickle
import pandas as pd
import numpy as np
import sys
import boto3
import s3fs


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)



categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    print("Reading in the data")

    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def apply_model(year,month):
    

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')

    print("Applying the Model...")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return df, y_pred

def save_results(df, y_pred,year,month):
    print("Saving results ....")

    df['ride_id']= f'{year:04d}/{month:02d}_'+df.index.astype('str')
    df_results = pd.concat(
        [df['ride_id'],pd.DataFrame(y_pred)],
        axis=1
        )
    df_results.rename(columns ={0:"predicted_duration"},inplace= True)


    output_file = f"s3://nyc-ride-pred/docker-output/year={year:04d}/month={month:02d}.parquet"

    df_results.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return df_results


def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    df, y_pred = apply_model(year,month)
    df_results = save_results(df,y_pred,year,month)
    pred_mean = round(np.mean(df_results['predicted_duration']),2)

    print(f"""The mean prediction duration for {month:02d}/{year}
           is {pred_mean}""")
    


if __name__ == "__main__":
    run()
