import pandas as pd
import numpy as np
import matplotlib as plt
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import Point
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


station_pos = pd.read_csv("weather/Minnesota Station location list.csv")
crop_df = pd.read_csv('agri/minnesota_county_yearly_agricultural_production.csv')


crs = {'init':'EPSG:4326'}
geometry = [Point(xy) for xy in zip(station_pos['Longitude'], station_pos['Latitude'])]
station_pos = gpd.GeoDataFrame(station_pos, 
                          crs = crs, 
                          geometry = geometry)



station_dict = {key: [] for key in station_pos['Code']}
station_data = os.listdir('weather/minnesota_daily')
path = 'weather/minnesota_daily/'
for i in station_data:
  name = i.split('.')[0]
  station_dict[name]  = pd.read_csv(path+i,header = None,names = ['date','tavg','tmin','tmax','prcp'])

crops = {}
for i in pd.unique(crop_df['County']):
  crops[i] = {}
  for j in pd.unique(crop_df['Commodity']):
    crops[i][j] = crop_df.loc[crop_df['County'] == i].loc[crop_df['Commodity'] == j].copy(deep=True).reset_index(drop=True)

target_dict = {}
target_data = os.listdir('weather/prediction_targets_daily/')
path = 'weather/prediction_targets_daily/'
for i in target_data:
  name = i.split('.')[0]
  target_dict[name]  = pd.read_csv(path+i,header = None,names = ['date','tavg','tmin','tmax','prcp'])
  if len(target_dict[name].index)<1:
    del target_dict[name]


county_ds = pd.read_csv("agri/minnesota_county_location.csv")
station_ds = pd.read_csv("weather/Minnesota Station location list.csv")
county_loc_ds = county_ds[['county_latitude','county_longitude']]
station_loc_ds = station_ds[['Latitude','Longitude']]
county_ds_abbr = county_ds.assign(county_abbr = lambda x: x['county_name'])
county_ds_abbr['county_abbr'] = [ s.replace(' County','').replace('Saint','St.').upper() for s in county_ds_abbr['county_abbr']]

imputer = IterativeImputer(missing_values = np.nan, initial_strategy = 'mean')

station_dict_imp = {}
c = ['tavg', 'tmin', 'tmax', 'prcp','year','month','day']
for i in station_dict:
  curr_df = station_dict[i].copy(deep=True)
  curr_df['date'] = pd.to_datetime(curr_df['date'])
  curr_df['year'] = pd.to_datetime(curr_df['date']).dt.year
  curr_df['month'] = pd.to_datetime(curr_df['date']).dt.month
  curr_df['day'] = pd.to_datetime(curr_df['date']).dt.day
  curr_df = curr_df.set_index('date')
  imputer.fit(curr_df[c].dropna())
  curr_df[c] = imputer.transform(curr_df[c])
  station_dict_imp[i] = curr_df 

#def impute(data,imputer):
#    c = ['tavg', 'tmin', 'tmax', 'prcp','year','month','day']
#    curr_df = data
#    curr_df['date'] = pd.to_datetime(curr_df['date'])
#    dmin = min(curr_df['date'])
#    dmax = max(curr_df['date'])
#    curr_df = curr_df.set_index('date')
#    new_dates = pd.date_range(start=dmin,end=dmax,freq='D')
#    curr_df =curr_df.reindex(new_dates)
#    curr_df['year'] = curr_df.index.year
#    curr_df['month'] = curr_df.index.month
#    curr_df['day'] = curr_df.index.day
#    curr_df[c] = imputer.transform(curr_df[c])
#    return curr_df

def impute(data,imputer):
    c = ['tavg', 'tmin', 'tmax', 'prcp','year','month','day']
    curr_df = data
    curr_df['date'] = pd.to_datetime(curr_df['date'])
    dmin = min(curr_df['date'])
    dmax = max(curr_df['date'])
    curr_df = curr_df.set_index('date')
    curr_df['year'] = curr_df.index.year
    curr_df['month'] = curr_df.index.month
    curr_df['day'] = curr_df.index.day
    curr_df[c] = imputer.transform(curr_df[c])
    return curr_df


def distance_matrix(a,b):
  
  res = np.zeros((len(a),len(b)))
  n=0
  for i in a.values:
    m=0
    for j in b.values:
      res[n][m] = np.linalg.norm(i - j)
      m+=1
    n +=1
  return res

def find_closest_n(a,b,a_cols,b_cols,n=2):
  mat=distance_matrix(a,b)
  #df = pd.DataFrame(mat,columns=b_cols)
  #index_m = np.zeros((len(a_cols),2))
  res = pd.DataFrame(index=range(len(a_cols)),columns=range(n))
  for i in range(len(mat)):
    for d in range(n):
      tempmat = mat[i]
      res[d][i] =b_cols[np.where(tempmat == np.min(tempmat))[0]].values[0]
      tempmat[np.where(tempmat == np.min(tempmat))[0]] = np.Inf
      #print(b_cols[np.where(tempmat == np.min(tempmat))[0]].values)
      #res[d][i] =np.where(tempmat == np.min(tempmat))[0]
  df = pd.concat([a_cols,res],axis = 1)
  return df

def model_trainer(features):
    train, test = train_test_split(features, test_size=0.2)
    X_train = pd.concat([train[list(train.columns)[0]],train[list(train.columns)[2:]]],axis=1).reset_index(drop = True)
    y_train = train['yield'].reset_index(drop = True)
    X_test = pd.concat([test[list(test.columns)[0]],test[list(test.columns)[2:]]],axis=1).reset_index(drop = True)
    y_test = test['yield'].reset_index(drop = True)
    models  = [ElasticNet(),SGDRegressor(),SVR(),BayesianRidge(),KernelRidge(),GradientBoostingRegressor(),LinearRegression()]
    maxScore = -5
    ideal_model = GradientBoostingRegressor()
    progression = 0
    for model in models:

        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        if maxScore<r2_score(y_test,y_pred):
            maxScore = r2_score(y_test,y_pred)
            ideal_model = model
        progression +=1
        print(f"Model {progression} of {len(models)} with the score of {r2_score(y_test,y_pred)}")
    print(f"The best model wass {type(ideal_model)} with the score of {maxScore}")
    return ideal_model

def sorter(n):
    closest_stations = find_closest_n(county_loc_ds,station_loc_ds,county_ds_abbr[county_ds_abbr.columns[-1]],station_ds[station_ds.columns[0]],n=n)
    relevant = closest_stations.copy(deep = True)
    l = []
    for i in range(len(relevant.values)):
        if(relevant['county_abbr'][i] not in crops.keys()):
            
            l.append(i)
    return relevant.drop(l).reset_index(drop = True)


def get_useful_weather_preds_for_commodity_and_crop(commodity = 'CORN',exact_crop = 'CORN, GRAIN',n=1):
    #n means the number of places our algo will look at when making the feature set
    relevant = sorter(n)
    harvest = []
    weather = []
    index = 0
    for ca in crops.keys():
        crps = crops[ca][commodity][crops[ca][commodity]['Crop']==exact_crop].dropna()
        years = crps['Year']
        crp_min_year = np.min(years)
        crp_max_year = np.max(years)

        if(len(years)!=0 ):
            #ha dict lenne:
            #corn[ca] = crps
            #ha lista lenne:
            harvest.append(crps)
            temp_l = []
            for i in range(n):
                temp_l.append(station_dict_imp[relevant[i][index]]
                [(station_dict_imp[relevant[i][index]]['year']>=crp_min_year)
                & (station_dict_imp[relevant[i][index]]['year']<=crp_max_year)])
                
            weather.append(temp_l)
        index +=1
    return weather,harvest    
u_keys = ['BARLEY',
 'CORN',
 'FLAXSEED',
 'OATS',
 'RYE',
 'SOYBEANS',
 'WHEAT',
 'WHEAT',
 'WHEAT',
 'WHEAT']
u_vals = ['BARLEY',
 'CORN, GRAIN',
 'FLAXSEED',
 'OATS',
 'RYE',
 'SOYBEANS',
 'WHEAT',
 'WHEAT, SPRING, (EXCL DURUM)',
 'WHEAT, SPRING, DURUM',
 'WHEAT, WINTER']

cols = ['year', 'yield', 'tmax', 'tavg','tmin','tmed','prcptavg','prcptmax']
def get_features(weather,crop):
    training_set = pd.DataFrame(columns=cols)
    feat_dict = dict.fromkeys(cols, 0)
    if(len(weather['year'])>0 and len(weather.values)>0):
        for i in crop['Year'][(crop['Year']>min(weather['year']))]:
            #i = 2007
            feat_dict['year'] = i
            feat_dict['yield'] = crop[(crop['Year'] == i)]['YIELD, MEASURED IN BU / ACRE'].values[0]
            feat_dict['tmax'] = np.max(weather[(weather['year']==i)]['tmax'])
            feat_dict['tmin'] = np.min(weather[(weather['year']==i)]['tmin'])
            feat_dict['tavg'] = np.average(weather[(weather['year']==i)]['tavg'])
            feat_dict['tmed'] = np.median(weather[(weather['year']==i)]['tavg'])
            feat_dict['prcptavg'] = np.average(weather[(weather['year']==i)]['prcp'])
            feat_dict['prcptmax'] = np.max(weather[(weather['year']==i)]['prcp'])
            featdf = pd.DataFrame([feat_dict])
            training_set = pd.concat([training_set,featdf], ignore_index=True)

    return training_set
target_cols = ['year' , 'tmax', 'tavg','tmin','tmed','prcptavg','prcptmax']
def get_target_features(weather):
    training_set = pd.DataFrame(columns=target_cols)
    feat_dict = dict.fromkeys(target_cols, 0)
    if(len(weather['year'])>0 and len(weather.values)>0):
        for i in weather['year'].unique():
            #i = 2007
            feat_dict['year'] = i
            feat_dict['tmax'] = np.max(weather[(weather['year']==i)]['tmax'])
            feat_dict['tmin'] = np.min(weather[(weather['year']==i)]['tmin'])
            feat_dict['tavg'] = np.average(weather[(weather['year']==i)]['tavg'])
            feat_dict['tmed'] = np.median(weather[(weather['year']==i)]['tavg'])
            feat_dict['prcptavg'] = np.average(weather[(weather['year']==i)]['prcp'])
            feat_dict['prcptmax'] = np.max(weather[(weather['year']==i)]['prcp'])
            featdf = pd.DataFrame([feat_dict])
            training_set = pd.concat([training_set,featdf], ignore_index=True)

    return training_set

def get_learning_feature_matrix(weather,crop,cols):
    end_df = pd.DataFrame(columns = cols)
    for i in range(len(crop)):
        for j in range(len(weather[0])):
            end_df = pd.concat([end_df, get_features(weather = weather[i][j],crop = crop[i])],ignore_index = True)
    return end_df.reset_index(drop = True)

def get_w_and_c_dicts():
    d_weather,d_commod = dict.fromkeys(u_vals),dict.fromkeys(u_vals)
    index = 0
    for i in u_vals:
        w,c = get_useful_weather_preds_for_commodity_and_crop(u_keys[index],i,3)
        if(len(w)>0 and len(c)>0):
            d_weather[i] =w
            d_commod[i] = c
        index +=1
    return d_weather,d_commod

def get_learning_feature_matrix_all():
    dw,dc = get_w_and_c_dicts()
    features_by_crops = dict.fromkeys(dw.keys())
    progress = 0
    print(f"Progress: {progress}/{len(dc)}",end='\r')
    for i in dc:
        features_by_crops[i] = get_learning_feature_matrix(dw[i],dc[i],cols).dropna().reset_index(drop = True).apply(pd.to_numeric)
        progress+=1
        print(f"Progress: {progress}/{len(dc)}",end='\r')
        
    return features_by_crops
def train_for_prediction(features):
    model = model_trainer(features)
    X_train = pd.concat([features[list(features.columns)[0]],features[list(features.columns)[2:]]],axis=1).reset_index(drop = True)
    y_train = features['yield'].reset_index(drop = True)
    model.fit(X_train,y_train)
    return model


def predict_all_targets_universal():
    d_feats = get_learning_feature_matrix_all()
    d_preds = dict.fromkeys(d_feats.keys())
    for feats in d_feats:
        model = train_for_prediction(d_feats[feats])
        target_names = list(target_dict.keys())
        preds = dict.fromkeys(target_names, [])
        length = len(target_names)
        counter = 0
        print(f"Progress:{counter}/{length}", end='\r')
        for i in target_dict:
            target_df = target_dict[i]
            target_df_imp = impute(target_df,imputer)
            p_feats = get_target_features(target_df_imp)
            pred_df = pd.DataFrame(data = model.predict(p_feats),index=p_feats['year'].unique().astype(np.int64))
            preds[i] = pred_df
            counter +=1
            print(f"Progress:{counter}/{length}", end='\r')
        print("\n")
        d_preds[feats] = preds
        print(feats,"done",end="\n")
    return d_preds

def write_all_preds_to_file(d_preds):
    cols = ['Target location','Year','Crop','Predicted yield (BU/acre)']
    result = pd.DataFrame(columns=cols)
    sprogress = 0
    for station in d_preds['BARLEY']:
        cprogress = 0
        for crop in d_preds:
            target_name = station
            prediction = d_preds[crop][station]
            appendict = dict.fromkeys(cols)
            for y in prediction.index:
                appendict['Target location'] = target_name
                appendict['Year'] =  y
                appendict['Crop'] = crop
                appendict['Predicted yield (BU/acre)'] =prediction.loc[y,0]
                result = pd.concat([result,pd.DataFrame(data=appendict,columns = cols, index=[0])])
        sprogress +=1
        print(f"station {sprogress} of {len(d_preds['BARLEY'])}" ,end = '\r')
    return result
del crops['OTHER (COMBINED) COUNTIES']
d_preds = predict_all_targets_universal()

allpreddf = write_all_preds_to_file(d_preds)

f = open('predictions_multicrop.csv','w')
f.write('"')
f.write('Target location,""Year"",""Crop"",""Predicted yield (BU/acre)""')
f.write('"\n')
for i in allpreddf.values:
    f.write('"')
    f.write(f'{i[0]},{i[1]},""{i[2]}"",{i[3]}')
    f.write('"\n')
f.close()