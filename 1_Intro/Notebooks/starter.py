import pickle
import pandas as pd
import click as ck

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df.ride_id
    # df_result['tpep_pickup_datetime'] = df.tpep_pickup_datetime
    # df_result['tpep_dropoff_datetime'] = df.tpep_dropoff_datetime
    # df_result["PULocationID"] = df.PULocationID
    # df_result["DOLocationID"] = df.DOLocationID
    # df_result["duration"] = df.duration
    df_result["duration_predicted"] = y_pred
    # df['prediction_error'] = df_result.duration_predicted - df_result.duration
    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
    )
# 1- 10.28  
# 2- 58  
# 3- jupyter nbconvert starter.ipynb --to python
# 4- 065e9673e24e0dc5113e2dd2b4ca30c9d8aa2fa90f4c0597241c93b63130d233
# 5- 12.76
# 6- 12.83

@ck.command()
@ck.option('--year', '-y', default='2022', help="The year of data you want to process", type=int)
@ck.option('--month', '-m', default='02', help='The month of data you want to process', type=int)
def main(year, month):
    print('loading model')
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet",
                    categorical=categorical)
    print('predicting')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print('saving predictions')

    # year = 2022
    # month = 2
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    print('predicted mean duration:', y_pred.mean())
    save_results(df, y_pred, 'predictions.parquet')

if __name__=='__main__':
    main()