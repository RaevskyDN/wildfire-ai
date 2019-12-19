# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import time
import xarray
import os
import sys
import itertools
import io
import json
from train_features import model_features
import datetime
from dateutil.relativedelta import relativedelta

import lightgbm as lgb
from sklearn.neighbors import KDTree,NearestNeighbors,KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")


DATASETS_PATH = os.environ.get('DATASETS_PATH', '../data/')


def get_day_of_year(date):
    doy = date.timetuple().tm_yday
    if date.year in (2012,2016,2020):
        if date >= datetime.date(date.year,3,1):
            doy -= 1
        elif date == datetime.date(date.year,2,29):
            doy = 366
    return doy

def calc_bearing(lon1,lat1,lon2,lat2):
    delta_lon = lon2 - lon1
    X = np.cos(math.radians(lat2)) * np.sin(math.radians(delta_lon))
    Y = np.cos(math.radians(lat1)) * np.sin(math.radians(lat2)) - np.sin(math.radians(lat1)) * np.cos(math.radians(lat2)) * np.cos(math.radians(delta_lon))
    bearing = math.atan2(X,Y)
    return bearing * 180/math.pi

def calc_distance(lon1,lat1,lon2,lat2):

    radius = 6371302  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d

class FeaturesMaker():
    def __init__(self):
        self.lat_min = 32.5
        self.lat_max = 75
        self.lon_min = 12.5
        self.lon_max = 182.5      
        self.base_levels = (1000,700,500,300,250,200,100,150,70,50)
        self.ncep_data = None
        self.lat_lons = np.array([x for x in itertools.product(np.arange(self.lon_min,self.lon_max+1,2.5),\
                                    np.arange(self.lat_min,self.lat_max+1,2.5))])
        self.kd_tree = KDTree(self.lat_lons)

    def make_spatial_features(self,df):
        self.full_train['date'] = pd.to_datetime(self.full_train['date'])
        self.full_train['timestamp'] = self.full_train['date'].apply(lambda x: time.mktime(x.timetuple()))
        df['timestamp'] = df['date'].apply(lambda x: time.mktime(x.timetuple()))
        
        self.kd_spatial_tree = KDTree(self.full_train[['longitude','latitude']])
        self.kd_spatial_time_tree = KDTree(self.full_train[['longitude','latitude','timestamp']])
        date_list = []
        start_date = datetime.date(2011,10,1)
        while start_date < datetime.date(2019,5,1):
            date_list.append(start_date)
            start_date += relativedelta(days=1)
        self.min_date_dict = pd.Series(np.nan,index=pd.to_datetime(date_list))
        self.max_date_dict = self.min_date_dict.copy()
        self.min_date_dict = self.full_train.set_index('fire_id').drop_duplicates('date',keep='first').reset_index().set_index('date')['fire_id'].\
                               combine(self.min_date_dict,max).fillna(method='bfill').fillna(-1).astype(int).to_dict()
        self.max_date_dict = self.full_train.set_index('fire_id').drop_duplicates('date',keep='first').reset_index().set_index('date')['fire_id'].\
                               combine(self.max_date_dict,max).fillna(method='ffill').fillna(-1).astype(int).to_dict()
        
        raw_features_df = df.set_index('fire_id')[['latitude','longitude','date','timestamp']].apply(self.return_nearest_previous_coord,axis=1)
        #only_spatial_features_df = self.make_only_spatial_features(df)
        only_spatial_features_df = self.time_series_features(df)
        return pd.merge(raw_features_df,only_spatial_features_df,left_index=True,right_index=True)

    def time_series_features(self,df):
        X = df[['latitude','longitude','fire_id']].copy()
        X['latitude_round'] = X['latitude'].round(2)
        X['longitude_round'] = X['longitude'].round(2)
        X.set_index('fire_id',inplace=True)
        self.full_train.set_index('fire_id',inplace=True)
        if set(self.full_train.index).intersection(X.index):
            dbs_df = self.full_train.copy()
        else:
            dbs_df = pd.concat([X,self.full_train])

        train_df = self.full_train[self.full_train['date'] < datetime.date(2019,1,1)].copy()
        train_df.sort_index(inplace=True)
        dbs_df.sort_index(inplace=True)
        X.sort_index(inplace=True)
        self.full_train.sort_index(inplace=True)
        #print(len(train_df))
        #print(len(self.full_train))
        #print(len(X))
        
        dbs = DBSCAN(eps=0.035,min_samples=1,n_jobs=4)
        dbs_df['label'] = dbs.fit_predict(dbs_df[['latitude','longitude']])
        train_df['label'] = dbs_df.loc[train_df.index,'label']
        X['label'] = dbs_df.loc[X.index,'label']
        train_df['count_035'] = train_df.groupby('label')['latitude'].transform('count')
        train_df['dist_035_latitude_mean'] = train_df.groupby('label')['latitude'].transform('mean')
        train_df['dist_035_longitude_mean'] = train_df.groupby('label')['longitude'].transform('mean')
        train_df['dist_035_square'] = (train_df.groupby('label')['latitude'].transform('max') - \
                                       train_df.groupby('label')['latitude'].transform('min'))*\
                                    (train_df.groupby('label')['longitude'].transform('max') - \
                                       train_df.groupby('label')['longitude'].transform('min'))
        for col in ['dist_035_latitude_mean','dist_035_longitude_mean','dist_035_square','count_035']:
            map_dict = train_df.drop_duplicates('label').set_index('label')[col]
            X[col] = dbs_df.loc[X.index,'label'].map(map_dict)
            nan_ind = X[X[col].isnull()].index
            if col == 'count_035':
                X.loc[nan_ind,col] = 0
            X.loc[nan_ind,col] = -1

        train_df['distance_from_035'] = train_df[['dist_035_longitude_mean','dist_035_latitude_mean','longitude','latitude']]\
                                    .apply(lambda x: calc_distance(x['dist_035_longitude_mean'],x['dist_035_latitude_mean'],x['longitude'],x['latitude']),axis=1)
        X['distance_from_035'] = X[['dist_035_longitude_mean','dist_035_latitude_mean','longitude','latitude']]\
                                    .apply(lambda x: calc_distance(x['dist_035_longitude_mean'],x['dist_035_latitude_mean'],x['longitude'],x['latitude']),axis=1)
        
        dbs = DBSCAN(eps=0.15,min_samples=1,n_jobs=4)
        dbs_df['label'] = dbs.fit_predict(dbs_df[['latitude','longitude']])
        train_df['label'] = dbs_df.loc[train_df.index,'label']
        X['label'] = dbs_df.loc[X.index,'label']
        train_df['count_15'] = train_df.groupby('label')['latitude'].transform('count')
        train_df['dist_15_square'] = (train_df.groupby('label')['latitude'].transform('max') - \
                                       train_df.groupby('label')['latitude'].transform('min'))*\
                                    (train_df.groupby('label')['longitude'].transform('max') - \
                                       train_df.groupby('label')['longitude'].transform('min'))
        for col in ['dist_15_square','count_15']:
            map_dict = train_df.drop_duplicates('label').set_index('label')[col]
            X[col] = dbs_df.loc[X.index,'label'].map(map_dict)
            nan_ind = X[X[col].isnull()].index
            if col == 'count_15':
                X.loc[nan_ind,col] = 0
            X.loc[nan_ind,col] = -1
        train_df['square_ratio'] = (train_df['dist_035_square']/train_df['dist_15_square']).fillna(0)
        X['square_ratio'] = (X['dist_035_square']/X['dist_15_square']).fillna(0)
        train_df['count_035_divide_count_15'] = (train_df['count_035']/train_df['count_15']).fillna(0)
        X['count_035_divide_count_15'] = (X['count_035']/X['count_15']).fillna(0)

        #train_df['latitude_big_round'] = train_df['latitude'].round(0)
        #train_df['longitude_big_round'] = train_df['longitude'].round(0)
        #X['latitude_big_round'] = X['latitude'].round(0)
        #X['longitude_big_round'] = X['longitude'].round(0)
        return X.drop(['label','latitude','longitude'],axis=1)

    def make_only_spatial_features(self,df):
        X = df[['latitude','longitude','fire_id']].copy()
        X['latitude_round'] = X['latitude'].round(2)
        X['longitude_round'] = X['longitude'].round(2)
        X.set_index('fire_id',inplace=True)
        self.full_train.set_index('fire_id',inplace=True)
        if set(self.full_train.index).intersection(X.index):
            df_dbscan = self.full_train.copy()
        else:
            df_dbscan = pd.concat([X,self.full_train[['latitude','longitude']]])
        print(df_dbscan.shape)
        dbs = DBSCAN(eps=0.035,min_samples=1,n_jobs=4)
        df_dbscan['label_035'] = dbs.fit_predict(df_dbscan[['latitude','longitude']])
        self.full_train['label_035'] = df_dbscan.loc[self.full_train.index,'label_035']
        self.full_train['count_035'] = self.full_train.groupby('label_035')['latitude'].transform('count')
        self.full_train['dist_035_latitude'] = self.full_train.groupby('label_035')['latitude'].transform('mean')
        self.full_train['dist_035_longitude'] = self.full_train.groupby('label_035')['longitude'].transform('mean')
        #kn_1 = KNeighborsClassifier(n_neighbors=1)
        #kn_1.fit(self.full_train[['longitude','latitude']],self.full_train['count_035'])

        #kn_1_label = KNeighborsClassifier(n_neighbors=1)
        #kn_1_label.fit(self.full_train[['longitude','latitude']],self.full_train['label'])
        latitude_035_dict = self.full_train.drop_duplicates('label_035',keep='first').set_index('label_035')['dist_035_latitude']
        longitude_035_dict = self.full_train.drop_duplicates('label_035',keep='first').set_index('label_035')['dist_035_longitude']

        dbs = DBSCAN(eps=0.055,min_samples=1,n_jobs=4)
        df_dbscan['label_055'] = dbs.fit_predict(df_dbscan[['latitude','longitude']])
        self.full_train['label_055'] = df_dbscan.loc[self.full_train.index,'label_055']
        self.full_train['count_055'] = self.full_train.groupby('label_055')['latitude'].transform('count')
        #kn_2 = KNeighborsClassifier(n_neighbors=1)
        #kn_2.fit(self.full_train[['longitude','latitude']],self.full_train['count_055'])
        
        dbs = DBSCAN(eps=0.15,min_samples=1,n_jobs=4)
        df_dbscan['label_15'] = dbs.fit_predict(df_dbscan[['latitude','longitude']])
        self.full_train['label_15'] = df_dbscan.loc[self.full_train.index,'label_15']
        self.full_train['count_15'] = self.full_train.groupby('label_15')['latitude'].transform('count')
        #kn_3 = KNeighborsClassifier(n_neighbors=1)
        #kn_3.fit(self.full_train[['longitude','latitude']],self.full_train['count_15'])
        
        X['dist_035_latitude'] = df_dbscan.loc[X.index,'label_035'].map(latitude_035_dict)
        #distances,_ = kn_1_label.kneighbors(X[['latitude','longitude']],return_distance=True)
        #X.loc[X.iloc[np.where(distances > 0.035)].index,'dist_035_latitude'] = X.loc[X.iloc[np.where(distances > 0.035)].index,'latitude']
        nan_ind = X[X['dist_035_latitude'].isnull()].index
        X.loc[nan_ind,'dist_035_latitude'] = X.loc[nan_ind,'latitude']
        
        X['dist_035_longitude'] = df_dbscan.loc[X.index,'label_035'].map(longitude_035_dict)
        #X.loc[X.iloc[np.where(distances > 0.035)].index,'dist_035_longitude'] = X.loc[X.iloc[np.where(distances > 0.035)].index,'longitude']
        nan_ind = X[X['dist_035_longitude'].isnull()].index
        X.loc[nan_ind,'dist_035_longitude'] = X.loc[nan_ind,'longitude']
        
        X['distance_from_035'] = X[['dist_035_longitude','dist_035_latitude','longitude','latitude']]\
                            .apply(lambda x: calc_distance(x['dist_035_longitude'],x['dist_035_latitude'],x['longitude'],x['latitude']),axis=1)

        count_035_dict = self.full_train.drop_duplicates('label_035').set_index('label_035')['count_035']
        X['count_035'] = df_dbscan.loc[X.index,'label_035'].map(count_035_dict)
        #distances,_ = kn_1.kneighbors(X[['latitude','longitude']],return_distance=True)
        X['count_035'] = X['count_035'].fillna(0)
        
        count_055_dict = self.full_train.drop_duplicates('label_055').set_index('label_055')['count_055']
        X['count_055'] = df_dbscan.loc[X.index,'label_055'].map(count_055_dict)
        #X['count_055'] = kn_2.predict(X[['longitude','latitude']])
        #distances,_ = kn_2.kneighbors(X[['latitude','longitude']],return_distance=True)
        X['count_055'] = X['count_055'].fillna(0)
        
        count_15_dict = self.full_train.drop_duplicates('label_15').set_index('label_15')['count_15']
        X['count_15'] = df_dbscan.loc[X.index,'label_15'].map(count_15_dict)
        #X['count_15'] = kn_3.predict(X[['longitude','latitude']])
        #distances,_ = kn_3.kneighbors(X[['latitude','longitude']],return_distance=True)
        #X.loc[X.iloc[np.where(distances > 0.15)].index,'count_15'] = 0
        X['count_15'] = X['count_15'].fillna(0)
        
        self.full_train['latitude_big_round'] = self.full_train['latitude'].round(0)
        self.full_train['longitude_big_round'] = self.full_train['longitude'].round(0)
        X['latitude_big_round'] = X['latitude'].round(0)
        X['longitude_big_round'] = X['longitude'].round(0)
        self.full_train['count_big_round'] = self.full_train.groupby(['latitude_big_round','longitude_big_round'])['latitude'].transform('count')
        self.full_train['label_latlon'] = self.full_train['latitude_big_round'].astype(str) + '-' + self.full_train['longitude_big_round'].astype(str)
        X['count_big_round'] = X['latitude_big_round'].astype(str) + '-' + X['longitude_big_round'].astype(str)
        count_dict = self.full_train.set_index('label_latlon')['count_big_round'].to_dict()
        X['count_big_round'] = X['count_big_round'].map(count_dict)
        X['count_big_round'] = X['count_big_round'].fillna(0)
        #for col in ['latitude_big_round','longitude_big_round','count_big_round']:
       #     X[col] = self.full_train.loc[X.index,col]
        
        self.full_train.reset_index(inplace=True)
        return X
        
    def divide_count_features(self,X):
        X['full_count_divide_count_035'] = X['full_count_15km']/X['count_035']
        X['full_count_divide_count_055'] = X['full_count_15km']/X['count_055']
        X['full_count_15_divide_full_count_5'] = X['full_count_5km']/X['full_count_15km']
        X['same_day_count_25km_divide_count_035'] = X['same_day_count_25km']/X['count_035']
        X['same_day_count_10km_divide_count_035'] = X['same_day_count_10km']/X['count_035']
        X['last_year_diff_week_count_divide_full_count_5'] = X['last_year_diff_week_count']/X['full_count_5km']
        for col in ['full_count_divide_count_035','full_count_divide_count_055','full_count_15_divide_full_count_5',\
               'same_day_count_25km_divide_count_035','same_day_count_10km_divide_count_035','last_year_diff_week_count_divide_full_count_5']:
            X[col] = X[col].fillna(-1)
    
    def return_nearest_previous_coord(self,row):
        indices,distances = self.kd_spatial_tree.query_radius([[row['longitude'],row['latitude']]],0.05,
                                        return_distance=True,sort_results=True)
        indices = indices[0][1:]
       #print(indices)
        
        count_3_years = 0
        
        count_last_year = 0
        
        lat_last_year_nearest = 0
        lon_last_year_nearest = 0
        distance_last_year_nearest = -1
        bearing_last_year_nearest = -200
        last_year_class = -1
        
        delta_week = relativedelta(days=7)
        delta_month = relativedelta(days=14)
        delta_year = relativedelta(years=1)
        start_year = row['date'] - delta_year
        condition = None
        condition_month = None
        condition_week = None
        last_year_condition = None
        last_year_condition_month = None
        n = 0
        end_year = row['date'].replace(year=2019,day=1,month=1)
        #print(start_year)
        while start_year >= datetime.date(2012,1,1):
            if n >= 3:
                break
            min_date_month = start_year - delta_month
            max_date_month = start_year + delta_month  
            min_date_week = start_year - delta_week
            max_date_week = start_year + delta_week   
            try:
                if condition is None:
                    #condition = ((indices <= self.max_date_dict[max_date]) & (indices >= self.min_date_dict[min_date]))
                    condition_month = ((indices <= self.max_date_dict[max_date_month]) & (indices >= self.min_date_dict[min_date_month]))
                    condition_week = ((indices <= self.max_date_dict[max_date_week]) & (indices >= self.min_date_dict[min_date_week]))
                    #last_year_condition = (indices <= self.max_date_dict[max_date]) & (indices >= self.min_date_dict[min_date])
                    last_year_condition_month = (indices <= self.max_date_dict[max_date_month]) & (indices >= self.min_date_dict[min_date_month])
                else:
                    #condition |= ((indices <= self.max_date_dict[max_date]) & (indices >= self.min_date_dict[min_date]))
                    condition_month = ((indices <= self.max_date_dict[max_date_month]) & (indices >= self.min_date_dict[min_date_month]))
                    condition_week = ((indices <= self.max_date_dict[max_date_week]) & (indices >= self.min_date_dict[min_date_week]))
            except:
                print(max_date,min_date)
                #print(condition)
                raise
            start_year -= delta_year
            n += 1
        #print(condition)
        #print(last_year_condition)
        #filtered_ind = np.where(last_year_condition)[0]
        #good_ind = indices[filtered_ind]
        #merge_dict = {'full_count_5km':len(indices)}
        if len(indices) > 0:
            all_condition = (indices <= self.max_date_dict[end_year]) 
            filtered_ind = np.where(all_condition)[0]
            merge_dict = {'5km_count':len(filtered_ind)}
        else:
            merge_dict = {'5km_count':0}
            
        filtered_ind = np.where(last_year_condition_month)[0]
        good_ind = indices[filtered_ind]
        merge_dict.update({'last_3_years_count':len(good_ind)})
        if len(good_ind) > 0:
            lat_last_year_nearest = self.full_train.iloc[good_ind[0]]['latitude']
            lon_last_year_nearest = self.full_train.iloc[good_ind[0]]['longitude']
            distance_last_year_nearest = calc_distance(row['longitude'],row['latitude'],
                                                       lon_last_year_nearest,lat_last_year_nearest)
            bearing_last_year_nearest = calc_bearing(row['longitude'],row['latitude'],
                                                      lon_last_year_nearest,lat_last_year_nearest)
            
        merge_dict.update({'diff_month_nearest_latitude':lat_last_year_nearest,
                          'diff_month_nearest_longitude':lon_last_year_nearest,
                          'diff_month_nearest_distance':distance_last_year_nearest,
                          'diff_month_count':len(good_ind),
                         })
        
        
        filtered_ind = np.where(condition_month)[0]
        good_ind = indices[filtered_ind]
        merge_dict.update({'diff_3_years_month_count':len(good_ind)})
        if len(good_ind) > 0:
            lat_last_year_nearest = self.full_train.iloc[good_ind[0]]['latitude']
            lon_last_year_nearest = self.full_train.iloc[good_ind[0]]['longitude']
            distance_last_year_nearest = calc_distance(row['longitude'],row['latitude'],
                                                       lon_last_year_nearest,lat_last_year_nearest)
            bearing_last_year_nearest = calc_bearing(row['longitude'],row['latitude'],
                                                      lon_last_year_nearest,lat_last_year_nearest)
            
        merge_dict.update({'diff_3_years_month_nearest_latitude':lat_last_year_nearest,
                          'diff_3_years_month_nearest_longitude':lon_last_year_nearest,
                          'diff_3_years_month_nearest_distance':distance_last_year_nearest,
                          'diff_3_years_month_count':len(good_ind),
                         })
        
        filtered_ind = np.where(condition_week)[0]
        good_ind = indices[filtered_ind]
        merge_dict.update({'last_year_diff_week_count':len(good_ind)})
        
        indices,distances = self.kd_spatial_time_tree.query_radius([[row['longitude'],row['latitude'],row['timestamp']]],
                                0.1,return_distance=True,sort_results=True)
        indices = indices[0][1:]
        
        count_same_day = 0
        lat_same_day_nearest = 0
        lon_same_day_nearest = 0
        distance_same_day_nearest = -1
        bearing_same_day_nearest = -200   
        if len(indices) > 0:
            lat_same_day_nearest = self.full_train.iloc[indices[0]]['latitude']
            lon_same_day_nearest = self.full_train.iloc[indices[0]]['longitude']
            distance_same_day_nearest = calc_distance(row['longitude'],row['latitude'],
                                                       lon_same_day_nearest,lat_same_day_nearest)
            bearing_same_day_nearest = calc_bearing(row['longitude'],row['latitude'],
                                                      lon_same_day_nearest,lat_same_day_nearest)
            count_same_day = len(indices)
        merge_dict.update({'same_day_nearest_latitude_10km':lat_same_day_nearest,
                          'same_day_nearest_longitude_10km':lon_same_day_nearest,
                          'same_day_nearest_distance_10km':distance_same_day_nearest,
                          'same_day_nearest_bearing_10km':bearing_same_day_nearest,
                          'same_day_count_10km':count_same_day
                         }) 
        
        indices,distances = self.kd_spatial_time_tree.query_radius([[row['longitude'],row['latitude'],row['timestamp']]],
                                0.25,return_distance=True,sort_results=True)
        indices = indices[0][1:]
        
        count_same_day = 0
        lat_same_day_nearest = 0
        lon_same_day_nearest = 0
        distance_same_day_nearest = -1
        bearing_same_day_nearest = -200   
        if len(indices) > 0:
            lat_same_day_nearest = self.full_train.iloc[indices[0]]['latitude']
            lon_same_day_nearest = self.full_train.iloc[indices[0]]['longitude']
            distance_same_day_nearest = calc_distance(row['longitude'],row['latitude'],
                                                       lon_same_day_nearest,lat_same_day_nearest)
            bearing_same_day_nearest = calc_bearing(row['longitude'],row['latitude'],
                                                      lon_same_day_nearest,lat_same_day_nearest)
            count_same_day = len(indices)
        merge_dict.update({'same_day_nearest_latitude_25km':lat_same_day_nearest,
                          'same_day_nearest_longitude_25km':lon_same_day_nearest,
                          'same_day_nearest_distance_25km':distance_same_day_nearest,
                          'same_day_nearest_bearing_25km':bearing_same_day_nearest,
                          'same_day_count_25km':count_same_day
                         })
                         
        indices,distances = self.kd_spatial_tree.query_radius([[row['longitude'],row['latitude']]],0.2,
                                        return_distance=True,sort_results=True)
        indices = indices[0][1:]
        if len(indices) > 0:
            all_condition = (indices <= self.max_date_dict[end_year])  
            filtered_ind = np.where(all_condition)[0]
            merge_dict.update({'20km_count':len(filtered_ind)})
            if merge_dict['20km_count'] > 0:
                merge_dict['5km_divide_20km'] = merge_dict['5km_count']/merge_dict['20km_count']
            else:
                merge_dict['5km_divide_20km'] = -1
            #except:
                #print(len(indices),indices,max_date_dict[end_year],len(filtered_ind),merge_dict['20km_count'],
                    # merge_dict['5km_count'])
                #raise
        else:
            merge_dict.update({'20km_count':0})
            merge_dict['5km_divide_20km'] = -1
        
        return pd.Series(merge_dict)
    
    def make_raw_features(self,df):
        start = time.time()
        #print('Making raw features...')
        basic_features = ['latitude','longitude']
        features_df = df[['fire_id','latitude','longitude','date']].copy()
        #features_df['weekday'] = features_df['date'].apply(lambda x: x.isocalendar()[2])
        features_df['day_of_year'] = features_df['date'].apply(get_day_of_year)
        features_df['number_of_week'] = features_df['date'].apply(lambda x: x.isocalendar()[1])
        features_df['month'] = features_df['date'].apply(lambda x: x.month)
        features_df['year'] = features_df['date'].apply(lambda x: x.year)
        #print('Making raw features done! Time {:.2f}'.format(time.time()-start))
        return features_df.drop('date',axis=1).set_index('fire_id')
    
    def make_temp_rhum_features(self,df):
        start = time.time()
        #print('Reading temperature-rhum-dataset...')
        self.read_temp_rhum_set()
        #print("Reading dataset done! Time {:.2f}".format(time.time()-start))
        start = time.time()
        #print('Making temperature-himidity features...')
        temp_rhum_features = self.extract_temp_rhum_features(df)
        #print('Making temperature-himidity features done! Time {:.2f}'.format(time.time()-start))
        return pd.DataFrame(temp_rhum_features).set_index('fire_id')
        
    def wind_direction(self,u,v,wind_module):
        try:
            wind_dir_trig_to = np.arctan2(u/wind_module,  v/wind_module)* 180/math.pi
        except:
            wind_dir_trig_to = 0    
        return wind_dir_trig_to
            
    def read_temp_rhum_set(self):        
        ncep_data = []
        for var in ('rhum','air'): #'uwnd','vwnd', 'rhum','trpp','srfpt'):
            #print(var)
            year = 2019
            dataset_filename = '{}/noaa/ncep/{}.{}.nc'.format(DATASETS_PATH,var, year)
            #dataset_filename = '{}.{}.nc'.format(var, year)
            array = xarray.open_dataset(dataset_filename)
            array = array.where((array.lat >= self.lat_min) &
                  (array.lat <= self.lat_max) & 
                  (array.lon <= self.lon_max) & 
                  (array.lon >= self.lon_min) & 
                  (array.level.isin(self.base_levels)),drop=True)
            ncep_data.append(array)
            my_max_date = datetime.date(2019,11,13)
            max_date = datetime.datetime.utcfromtimestamp(int((array.time.max().item(0)/10e8)+3600*24)).date()
            print('{} {}'.format(var,max_date))
            if max_date < my_max_date:
                dataset_filename = '{}.{}.nc'.format(var, year)
                array = xarray.open_dataset(dataset_filename)
                array = array.where((array.lat >= self.lat_min) &
                  (array.lat <= self.lat_max) & 
                  (array.lon <= self.lon_max) & 
                  (array.lon >= self.lon_min) & 
                  (array.level.isin(self.base_levels)),drop=True)
                array = array.sel(time=slice(max_date.strftime('%Y-%m-%d'),'2019-11-13'))
                ncep_data.append(array)
        full_array = xarray.merge(ncep_data)
        self.ncep_data = full_array
        
    def extract_temp_rhum_features(self,df):   
        features_df = df[['fire_id','longitude','latitude','lat','lon','lat_near_2','lon_near_2','lat_near_3','lon_near_3','lat_near_4','lon_near_4',
                                  'time']].copy()
        features_df['bearing_from_nearest'] = features_df[['lon','lat','longitude','latitude']]\
                                   .apply(lambda x: calc_bearing(x['lon'],x['lat'],x['longitude'],x['latitude']),axis=1)
        features_df['distance_from_nearest'] = features_df[['lon','lat','longitude','latitude']]\
                                   .apply(lambda x: calc_distance(x['lon'],x['lat'],x['longitude'],x['latitude']),axis=1)
        
        for level in self.base_levels:
            point_df = self.ncep_data.sel(
                level=level,
            ).to_dataframe().reset_index().rename(columns={'air':'temperature_level{}'.format(level),
                                                          'rhum':'humidity_level{}'.format(level)})
            air = 'temperature_level{}'.format(level)
            rhum = 'humidity_level{}'.format(level)
            
            if level == self.base_levels[0]:
                print(point_df['time'].max(),point_df['time'].min())

            p21d = point_df.groupby(['lat','lon']).rolling(window=21,min_periods=1,on='time')[[air,rhum]].mean().reset_index().\
                  rename(columns={air:'p21d-t_level{}'.format(level),rhum:'p21d-h_level{}'.format(level)})
            
            p35d = point_df.groupby(['lat','lon']).rolling(window=35,min_periods=1,on='time')[[air,rhum]].mean().reset_index().\
                  rename(columns={air:'p35d-t_level{}'.format(level),rhum:'p35d-h_level{}'.format(level)})
            p35d['time'] = p35d['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            p7d = point_df.groupby(['lat','lon']).rolling(window=7,min_periods=1,on='time')[[air,rhum]].mean().reset_index().\
                  rename(columns={air:'p7d-t_level{}'.format(level),rhum:'p7d-h_level{}'.format(level)})
            
            p14d = point_df.groupby(['lat','lon']).rolling(window=14,min_periods=1,on='time')[[air,rhum]].mean().reset_index().\
                  rename(columns={air:'p14d-t_level{}'.format(level),rhum:'p14d-h_level{}'.format(level)})
            p14d['time'] = p14d['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            #point_df_diff = point_df.groupby(['lat','lon'])[[air,rhum]].diff().fillna(0).reset_index().\
            #             rename(columns={air:'t_prev_diff_level{}'.format(level),rhum:'h_prev_diff_level{}'.format(level)})
            #point_df_diff = point_df_diff.merge(point_df[['time','lat','lon']],left_index=True,right_index=True)
            #point_df_diff['time'] = point_df_diff['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            lat_lon_df = point_df[point_df['time'] >= datetime.date(2019,11,1)].copy()
            lat_lon_df.drop_duplicates(subset=['lat','lon'],keep='first',inplace=True)
            lat_lon_df.rename(columns={'temperature_level{}'.format(level):'first_temperature_level{}'.format(level),
                                      'humidity_level{}'.format(level):'first_humidity_level{}'.format(level)},inplace=True)
            lat_lon_df['time'] = lat_lon_df['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            lat_lon_df_2 = point_df[point_df['time'] >= datetime.date(2019,9,1)].copy()
            lat_lon_df_2.drop_duplicates(subset=['lat','lon'],keep='first',inplace=True)
            lat_lon_df_2.rename(columns={'temperature_level{}'.format(level):'first_20190901_temperature_level{}'.format(level),
                                      'humidity_level{}'.format(level):'first_20190901_humidity_level{}'.format(level)},inplace=True)
            lat_lon_df_2['time'] = lat_lon_df_2['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            lat_lon_df_21 = p21d[p21d['time'] >= datetime.date(2019,11,1)].copy()
            lat_lon_df_21.drop_duplicates(subset=['lat','lon'],keep='first',inplace=True)
            lat_lon_df_21.rename(columns={'p21d-t_level{}'.format(level):'first_21d_temperature_level{}'.format(level),
                                      'p21d-h_level{}'.format(level):'first_21d_humidity_level{}'.format(level)},inplace=True)
            lat_lon_df_21['time'] = lat_lon_df_21['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            lat_lon_df_7 = p7d[p7d['time'] >= datetime.date(2019,10,1)].copy()
            lat_lon_df_7.drop_duplicates(subset=['lat','lon'],keep='first',inplace=True)
            lat_lon_df_7.rename(columns={'p7d-t_level{}'.format(level):'first_7d_temperature_level{}'.format(level),
                                      'p7d-h_level{}'.format(level):'first_7d_humidity_level{}'.format(level)},inplace=True)
            lat_lon_df_7['time'] = lat_lon_df_7['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            p21d['time'] = p21d['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            p7d['time'] = p7d['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            point_df['time'] = point_df['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            features_df = features_df.merge(p14d[['lat','lon','time','p14d-t_level{}'.format(level),'p14d-h_level{}'.format(level)]],
                                on=['lat','lon','time'])
            features_df = features_df.merge(p7d[['lat','lon','time','p7d-t_level{}'.format(level),'p7d-h_level{}'.format(level)]],
                                on=['lat','lon','time'])
            features_df = features_df.merge(p21d[['lat','lon','time','p21d-t_level{}'.format(level),'p21d-h_level{}'.format(level)]],
                                on=['lat','lon','time'])
            features_df = features_df.merge(p35d[['lat','lon','time','p35d-t_level{}'.format(level),'p35d-h_level{}'.format(level)]],
                                on=['lat','lon','time'])
            features_df = features_df.merge(point_df[['lat','lon','time','temperature_level{}'.format(level),'humidity_level{}'.format(level)]],
                                on=['lat','lon','time'])
            #features_df = features_df.merge(point_df_diff[['lat','lon','time','t_prev_diff_level{}'.format(level),
            #                            'h_prev_diff_level{}'.format(level)]],on=['lat','lon','time'])
            
            features_df = features_df.merge(lat_lon_df_2[['lat','lon','first_20190901_temperature_level{}'.format(level),
                                            'first_20190901_humidity_level{}'.format(level)]],on=['lat','lon'])
            features_df = features_df.merge(lat_lon_df[['lat','lon','first_temperature_level{}'.format(level),
                                            'first_humidity_level{}'.format(level)]],on=['lat','lon'])
            features_df = features_df.merge(lat_lon_df_21[['lat','lon','first_21d_temperature_level{}'.format(level),
                                            'first_21d_humidity_level{}'.format(level)]],on=['lat','lon'])
            features_df = features_df.merge(lat_lon_df_7[['lat','lon','first_7d_temperature_level{}'.format(level),
                                            'first_7d_humidity_level{}'.format(level)]],on=['lat','lon'])
            features_df['p14d_diff-t_level{}'.format(level)] = (features_df['temperature_level{}'.format(level)] - \
                                               features_df['p14d-t_level{}'.format(level)])
            features_df['p14d_diff-h_level{}'.format(level)] = (features_df['humidity_level{}'.format(level)] - \
                                               features_df['p14d-h_level{}'.format(level)])
            #features_df['p35d_diff-t_level{}'.format(level)] = (features_df['temperature_level{}'.format(level)] - \
             #                                  features_df['p35d-t_level{}'.format(level)])
            #features_df['p35d_diff-h_level{}'.format(level)] = (features_df['humidity_level{}'.format(level)] - \
             #                                  features_df['p35d-h_level{}'.format(level)])
            
            for nearest in [2,3,4]:
                features_df = features_df.merge(point_df[['lat','lon','time','temperature_level{}'.format(level),'humidity_level{}'.format(level)]].\
                                              rename(columns={'temperature_level{}'.format(level):'temperature_near_{}_level_{}'.format(nearest,level),
                                                              'humidity_level{}'.format(level):'humidity_near_{}_level_{}'.format(nearest,level),
                                                             'lat':'lat_near_{}'.format(nearest),
                                                             'lon':'lon_near_{}'.format(nearest)}),\
                                on=['lat_near_{}'.format(nearest),'lon_near_{}'.format(nearest),'time'])
                
                
                features_df = features_df.merge(lat_lon_df[['lat','lon','first_temperature_level{}'.format(level),
                                                          'first_humidity_level{}'.format(level)]].\
                    rename(columns={'first_temperature_level{}'.format(level):'first_temperature_near_{}_level_{}'.format(nearest,level),
                    'first_humidity_level{}'.format(level):'first_humidity_near_{}_level_{}'.format(nearest,level),
                    'lat':'lat_near_{}'.format(nearest),
                    'lon':'lon_near_{}'.format(nearest)}),\
                                on=['lat_near_{}'.format(nearest),'lon_near_{}'.format(nearest)]) 
                features_df['distance_{}'.format(nearest)] = features_df[['lon_near_{}'.format(nearest),'lat_near_{}'.format(nearest),'longitude','latitude']]\
                      .apply(lambda x: calc_distance(x['lon_near_{}'.format(nearest)],x['lat_near_{}'.format(nearest)],x['longitude'],x['latitude']),axis=1)
            
                
            
            features_df['harmonic_mean-t_level{}'.format(level)] = (4/((1/features_df['temperature_level{}'.format(level)]) +\
                                                           (1/features_df['temperature_near_2_level_{}'.format(level)]) +\
                                                           (1/features_df['temperature_near_3_level_{}'.format(level)]) +\
                                                           (1/features_df['temperature_near_4_level_{}'.format(level)])))
            
            features_df['harmonic_mean-h_level{}'.format(level)] = (4/((1/features_df['humidity_level{}'.format(level)]) +\
                                                           (1/features_df['humidity_near_2_level_{}'.format(level)]) +\
                                                           (1/features_df['humidity_near_3_level_{}'.format(level)]) +\
                                                           (1/features_df['humidity_near_4_level_{}'.format(level)])))

            features_df['dist_weighted_mean-t_level{}'.format(level)] = (features_df['temperature_level{}'.format(level)] *\
                                                           features_df['distance_from_nearest'] +\
                                                           features_df['temperature_near_2_level_{}'.format(level)] *\
                                                           features_df['distance_2'] +\
                                                           features_df['temperature_near_3_level_{}'.format(level)] *\
                                                           features_df['distance_3']+\
                                                           features_df['temperature_near_4_level_{}'.format(level)] *\
                                                           features_df['distance_4'])/(features_df['distance_from_nearest'] + \
                                                    features_df['distance_2']+features_df['distance_3']+features_df['distance_4'])
            
            features_df['dist_weighted_mean-h_level{}'.format(level)] = (features_df['humidity_level{}'.format(level)] *\
                                                           features_df['distance_from_nearest'] +\
                                                           features_df['humidity_near_2_level_{}'.format(level)] *\
                                                           features_df['distance_2'] +\
                                                           features_df['humidity_near_3_level_{}'.format(level)] *\
                                                           features_df['distance_3']+\
                                                           features_df['humidity_near_4_level_{}'.format(level)] *\
                                                           features_df['distance_4'])/(features_df['distance_from_nearest'] + \
                                                    features_df['distance_2']+features_df['distance_3']+features_df['distance_4'])
            
            features_df['mean-t_level{}'.format(level)] = (features_df['temperature_level{}'.format(level)] +\
                                                           features_df['temperature_near_2_level_{}'.format(level)] +\
                                                           features_df['temperature_near_3_level_{}'.format(level)] +\
                                                           features_df['temperature_near_4_level_{}'.format(level)])/4
            
            features_df['mean-h_level{}'.format(level)] = (features_df['humidity_level{}'.format(level)] +\
                                                           features_df['humidity_near_2_level_{}'.format(level)] +\
                                                           features_df['humidity_near_3_level_{}'.format(level)] +\
                                                           features_df['humidity_near_4_level_{}'.format(level)])/4
            
            
            features_df['first_mean-t_level{}'.format(level)] = (features_df['first_temperature_level{}'.format(level)] +\
                                                           features_df['first_temperature_near_2_level_{}'.format(level)] +\
                                                           features_df['first_temperature_near_3_level_{}'.format(level)] +\
                                                           features_df['first_temperature_near_4_level_{}'.format(level)])/4
            
            features_df['first_mean-h_level{}'.format(level)] = (features_df['first_humidity_level{}'.format(level)] +\
                                                           features_df['first_humidity_near_2_level_{}'.format(level)] +\
                                                           features_df['first_humidity_near_3_level_{}'.format(level)] +\
                                                           features_df['first_humidity_near_4_level_{}'.format(level)])/4
            
            features_df['dist_weighted_first_mean-t_level{}'.format(level)] = (features_df['first_temperature_level{}'.format(level)] *\
                                                           features_df['distance_from_nearest'] +\
                                                           features_df['first_temperature_near_2_level_{}'.format(level)] *\
                                                           features_df['distance_2'] +\
                                                           features_df['first_temperature_near_3_level_{}'.format(level)] *\
                                                           features_df['distance_3'] +\
                                                           features_df['first_temperature_near_4_level_{}'.format(level)] *\
                                                           features_df['distance_4'])/(features_df['distance_from_nearest'] + \
                                                    features_df['distance_2']+features_df['distance_3']+features_df['distance_4'])
            
            features_df['dist_weighted_irst_mean-h_level{}'.format(level)] = (features_df['first_humidity_level{}'.format(level)] *\
                                                           features_df['distance_from_nearest'] +\
                                                           features_df['first_humidity_near_2_level_{}'.format(level)] *\
                                                           features_df['distance_2'] +\
                                                           features_df['first_humidity_near_3_level_{}'.format(level)] *\
                                                           features_df['distance_3'] +\
                                                           features_df['first_humidity_near_4_level_{}'.format(level)] *\
                                                           features_df['distance_4'])/(features_df['distance_from_nearest'] + \
                                                    features_df['distance_2']+features_df['distance_3']+features_df['distance_4'])
            
            features_df['diff_mean-t_level{}'.format(level)] = features_df['temperature_level{}'.format(level)] - features_df['mean-t_level{}'.format(level)]
            features_df['diff_first_mean-t_level{}'.format(level)] = features_df['temperature_level{}'.format(level)] - features_df['first_mean-t_level{}'.format(level)]
            features_df['diff_first_mean-first_t_level{}'.format(level)] = features_df['first_temperature_level{}'.format(level)] - features_df['first_mean-t_level{}'.format(level)]
            features_df['diff_first-t_level{}'.format(level)] = features_df['temperature_level{}'.format(level)] - features_df['first_temperature_level{}'.format(level)]
            features_df['diff_mean-h_level{}'.format(level)] = features_df['humidity_level{}'.format(level)] - features_df['mean-h_level{}'.format(level)]
            features_df['diff_first_mean-h_level{}'.format(level)] = features_df['humidity_level{}'.format(level)] - features_df['first_mean-h_level{}'.format(level)]
            features_df['diff_first-h_level{}'.format(level)] = features_df['humidity_level{}'.format(level)] - features_df['first_humidity_level{}'.format(level)]
            features_df['diff_first_mean-first_h_level{}'.format(level)] = features_df['first_humidity_level{}'.format(level)] - features_df['first_mean-h_level{}'.format(level)]
            #features_df['diff_first-h-t_level{}'.format(level)] = features_df['diff_first-h_level{}'.format(level)] * features_df['diff_first-t_level{}'.format(level)]
            #features_df['h-t_level{}'.format(level)] = features_df['temperature_level{}'.format(level)] * features_df['humidity_level{}'.format(level)]
        for level_1,level_2 in [(1000,700),(100,50),(1000,50),(1000,300),(70,50),(150,50)]:
            features_df['temperature_level_{}_{}_diff'.format(level_1,level_2)] = features_df['temperature_level{}'.format(level_1)] -  features_df['temperature_level{}'.format(level_2)]
            features_df['p14d-t_level_{}_{}_diff'.format(level_1,level_2)] = features_df['p14d-t_level{}'.format(level_1)] -  features_df['p14d-t_level{}'.format(level_2)]
            features_df['first_mean-t_level_{}_{}_diff'.format(level_1,level_2)] = features_df['first_mean-t_level{}'.format(level_1)] -  features_df['first_mean-t_level{}'.format(level_2)]
            features_df['first-t_level_{}_{}_diff'.format(level_1,level_2)] = features_df['first_temperature_level{}'.format(level_1)] -  features_df['first_temperature_level{}'.format(level_2)]
        for level_1,level_2 in [(1000,700),(500,300),(1000,300),(700,300)]:
            features_df['humidity_level_{}_{}_diff'.format(level_1,level_2)] = features_df['humidity_level{}'.format(level_1)] -  features_df['humidity_level{}'.format(level_2)]
            features_df['p14d-h_level_{}_{}_diff'.format(level_1,level_2)] = features_df['p14d-h_level{}'.format(level_1)] -  features_df['p14d-h_level{}'.format(level_2)]
            features_df['first_mean-h_level_{}_{}_diff'.format(level_1,level_2)] = features_df['first_mean-h_level{}'.format(level_1)] -  features_df['first_mean-h_level{}'.format(level_2)]
            features_df['first-h_level_{}_{}_diff'.format(level_1,level_2)] = features_df['first_humidity_level{}'.format(level_1)] -  features_df['first_humidity_level{}'.format(level_2)]
        return features_df.drop(['lat_near_2','lon_near_2','lat_near_3','lon_near_3','lat_near_4','lon_near_4',
                                 'time','latitude','longitude',
                            'temperature_near_2_level_{}'.format(level),'temperature_near_3_level_{}'.format(level),
                                'temperature_near_4_level_{}'.format(level),'humidity_near_3_level_{}'.format(level),
                                'humidity_near_3_level_{}'.format(level),'humidity_near_3_level_{}'.format(level),
                                'first_temperature_near_2_level_{}'.format(level),'first_temperature_near_3_level_{}'.format(level),
                                'first_temperature_near_4_level_{}'.format(level),'first_humidity_near_3_level_{}'.format(level),
                                'first_humidity_near_3_level_{}'.format(level),'first_humidity_near_3_level_{}'.format(level)],
                                 axis=1,errors='ignore')
                       
    
    def extract_winds_features(self,df):   
        features_df = df[['fire_id','longitude','latitude','lat','lon','lat_near_2','lon_near_2','lat_near_3','lon_near_3','lat_near_4','lon_near_4',
                                  'time']].copy()
        
        for level in self.base_levels:
            point_df = self.ncep_data.sel(
                level=level,
            ).to_dataframe().reset_index().rename(columns={'uwnd':'uwind_level{}'.format(level),
                                                          'vwnd':'vwind_level{}'.format(level)})
            uwnd = 'uwind_level{}'.format(level)
            vwnd = 'vwind_level{}'.format(level)
            
            if level == self.base_levels[0]:
                print(point_df['time'].max(),point_df['time'].min())
            
            p21d = point_df.groupby(['lat','lon']).rolling(window=21,min_periods=1,on='time')[[uwnd,vwnd]].mean().reset_index().\
                  rename(columns={uwnd:'p21d-uw_level{}'.format(level),vwnd:'p21d-vw_level{}'.format(level)})
            
            p35d = point_df.groupby(['lat','lon']).rolling(window=35,min_periods=1,on='time')[[uwnd]].mean().reset_index().\
                  rename(columns={uwnd:'p35d-uw_level{}'.format(level)})
            p35d['time'] = p35d['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            #p50d = point_df.groupby(['lat','lon']).rolling(window=50,min_periods=1,on='time')[[uwnd]].mean().reset_index().\
            #      rename(columns={uwnd:'p50d-uw_level{}'.format(level)})
            #p50d['time'] = p50d['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            p7d = point_df.groupby(['lat','lon']).rolling(window=7,min_periods=1,on='time')[[uwnd,vwnd]].mean().reset_index().\
                  rename(columns={uwnd:'p7d-uw_level{}'.format(level),vwnd:'p7d-vw_level{}'.format(level)})
            
            p14d = point_df.groupby(['lat','lon']).rolling(window=14,min_periods=1,on='time')[[uwnd]].mean().reset_index().\
                  rename(columns={uwnd:'p14d-uw_level{}'.format(level)})
            p14d['time'] = p14d['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            point_df_diff = point_df.groupby(['lat','lon'])[[uwnd]].diff().fillna(0).reset_index().\
                         rename(columns={uwnd:'uw_prev_diff_level{}'.format(level)})
            point_df_diff = point_df_diff.merge(point_df[['time','lat','lon']],left_index=True,right_index=True)
            point_df_diff['time'] = point_df_diff['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            lat_lon_df = point_df[point_df['time'] >= datetime.date(2019,11,1)].copy()
            lat_lon_df.drop_duplicates(subset=['lat','lon'],keep='first',inplace=True)
            lat_lon_df.rename(columns={'uwind_level{}'.format(level):'first_uwind_level{}'.format(level),
                                      'vwind_level{}'.format(level):'first_vwind_level{}'.format(level)},inplace=True)
            lat_lon_df['time'] = lat_lon_df['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            lat_lon_df_2 = point_df[point_df['time'] >= datetime.date(2019,9,1)].copy()
            lat_lon_df_2.drop_duplicates(subset=['lat','lon'],keep='first',inplace=True)
            lat_lon_df_2.rename(columns={'uwind_level{}'.format(level):'first_20190901_uwind_level{}'.format(level),
                                      'vwind_level{}'.format(level):'first_20190901_vwind_level{}'.format(level)},inplace=True)
            lat_lon_df_2['time'] = lat_lon_df_2['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            lat_lon_df_21 = p21d[p21d['time'] >= datetime.date(2019,11,1)].copy()
            lat_lon_df_21.drop_duplicates(subset=['lat','lon'],keep='first',inplace=True)
            lat_lon_df_21.rename(columns={'p21d-uw_level{}'.format(level):'first_21d_uwind_level{}'.format(level),
                                      'p21d-vw_level{}'.format(level):'first_21d_vwind_level{}'.format(level)},inplace=True)
            lat_lon_df_21['time'] = lat_lon_df_21['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            lat_lon_df_7 = p7d[p7d['time'] >= datetime.date(2019,10,1)].copy()
            lat_lon_df_7.drop_duplicates(subset=['lat','lon'],keep='first',inplace=True)
            lat_lon_df_7.rename(columns={'p7d-uw_level{}'.format(level):'first_7d_uwind_level{}'.format(level),
                                      'p7d-vw_level{}'.format(level):'first_7d_vwind_level{}'.format(level)},inplace=True)
            lat_lon_df_7['time'] = lat_lon_df_7['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            
            point_df['time'] = point_df['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            p21d['time'] = p21d['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            p7d['time'] = p7d['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
                

            features_df = features_df.merge(p14d[['lat','lon','time','p14d-uw_level{}'.format(level)]],
                                on=['lat','lon','time'])
            features_df = features_df.merge(p7d[['lat','lon','time','p7d-uw_level{}'.format(level)]],
                                on=['lat','lon','time'])
            features_df = features_df.merge(p21d[['lat','lon','time','p21d-uw_level{}'.format(level)]],
                                on=['lat','lon','time'])
            features_df = features_df.merge(p35d[['lat','lon','time','p35d-uw_level{}'.format(level)]],
                                on=['lat','lon','time'])
            features_df = features_df.merge(point_df[['lat','lon','time','uwind_level{}'.format(level)]],
                                on=['lat','lon','time'])
            #features_df = features_df.merge(point_df_diff[['lat','lon','time',
            #                        'uw_prev_diff_level{}'.format(level)]],on=['lat','lon','time'])
            
            features_df = features_df.merge(lat_lon_df[['lat','lon','first_uwind_level{}'.format(level),
                                                       'first_vwind_level{}'.format(level)]],
                                            on=['lat','lon'])
            
            features_df['p14d_diff-uw_level{}'.format(level)] = (features_df['uwind_level{}'.format(level)] - \
                                               features_df['p14d-uw_level{}'.format(level)])
            #features_df['p35d_diff-uw_level{}'.format(level)] = (features_df['uwind_level{}'.format(level)] - \
            #                                   features_df['p35d-uw_level{}'.format(level)])

            for nearest in [2,3,4]:
                features_df = features_df.merge(point_df[['lat','lon','time','uwind_level{}'.format(level)]].\
                                              rename(columns={'uwind_level{}'.format(level):'uwind_near_{}_level_{}'.format(nearest,level),
                                                             'lat':'lat_near_{}'.format(nearest),
                                                             'lon':'lon_near_{}'.format(nearest)}),
                                on=['lat_near_{}'.format(nearest),'lon_near_{}'.format(nearest),'time'])
                
                features_df = features_df.merge(lat_lon_df[['lat','lon','first_uwind_level{}'.format(level),
                                                           'first_vwind_level{}'.format(level)]].\
                    rename(columns={'first_uwind_level{}'.format(level):'first_uwind_near_{}_level_{}'.format(nearest,level),
                                    'first_vwind_level{}'.format(level):'first_vwind_near_{}_level_{}'.format(nearest,level),
                    'lat':'lat_near_{}'.format(nearest),
                    'lon':'lon_near_{}'.format(nearest)}),\
                                on=['lat_near_{}'.format(nearest),'lon_near_{}'.format(nearest)]) 

              
            
            features_df['harmonic_mean-uw_level{}'.format(level)] = (4/((1/features_df['uwind_level{}'.format(level)]) +\
                                                           (1/features_df['uwind_near_2_level_{}'.format(level)]) +\
                                                           (1/features_df['uwind_near_3_level_{}'.format(level)]) +\
                                                           (1/features_df['uwind_near_4_level_{}'.format(level)])))
            
            #features_df['harmonic_mean-vw_level{}'.format(level)] = (4/((1/features_df['vwind_level{}'.format(level)]) +\
            #                                               (1/features_df['vwind_near_2_level_{}'.format(level)]) +\
            #                                               (1/features_df['vwind_near_3_level_{}'.format(level)]) +\
            #                                               (1/features_df['vwind_near_4_level_{}'.format(level)])))
            
            features_df['mean-uw_level{}'.format(level)] = (features_df['uwind_level{}'.format(level)] +\
                                                           features_df['uwind_near_2_level_{}'.format(level)] +\
                                                           features_df['uwind_near_3_level_{}'.format(level)] +\
                                                           features_df['uwind_near_4_level_{}'.format(level)])/4
            
            #features_df['mean-vw_level{}'.format(level)] = (features_df['vwind_level{}'.format(level)] +\
            #                                               features_df['vwind_near_2_level_{}'.format(level)] +\
            #                                               features_df['vwind_near_3_level_{}'.format(level)] +\
            #                                               features_df['vwind_near_4_level_{}'.format(level)])/4
            
            features_df['first_mean-uw_level{}'.format(level)] = (features_df['first_uwind_level{}'.format(level)] +\
                                                           features_df['first_uwind_near_2_level_{}'.format(level)] +\
                                                           features_df['first_uwind_near_3_level_{}'.format(level)] +\
                                                           features_df['first_uwind_near_4_level_{}'.format(level)])/4
            features_df['first_mean-vw_level{}'.format(level)] = (features_df['first_vwind_level{}'.format(level)] +\
                                                           features_df['first_vwind_near_2_level_{}'.format(level)] +\
                                                           features_df['first_vwind_near_3_level_{}'.format(level)] +\
                                                           features_df['first_vwind_near_4_level_{}'.format(level)])/4
            
            features_df['first_wind_module_level{}'.format(level)] = (features_df['first_vwind_level{}'.format(level)] ** 2 + \
                                                               features_df['first_uwind_level{}'.format(level)] ** 2)**0.5
            features_df['first_wind_direction_level{}'.format(level)] = features_df[['first_uwind_level{}'.format(level),
                                                'first_vwind_level{}'.format(level),
                                                'first_wind_module_level{}'.format(level)]].\
                                                 apply(lambda x:self.wind_direction(x['first_uwind_level{}'.format(level)],
                                                                x['first_vwind_level{}'.format(level)],
                                                                        x['first_wind_module_level{}'.format(level)]),axis=1)
            
            features_df['diff_mean-uw_level{}'.format(level)] = features_df['uwind_level{}'.format(level)] - features_df['mean-uw_level{}'.format(level)]
            features_df['diff_first_mean-uw_level{}'.format(level)] = features_df['uwind_level{}'.format(level)] - features_df['first_mean-uw_level{}'.format(level)]
            features_df['diff_first-uw_level{}'.format(level)] = features_df['uwind_level{}'.format(level)] - features_df['first_uwind_level{}'.format(level)]
            features_df['diff_first_mean-first_uw_level{}'.format(level)] = features_df['first_uwind_level{}'.format(level)] - features_df['first_mean-uw_level{}'.format(level)]
            features_df['diff_first_mean-first_vw_level{}'.format(level)] = features_df['first_vwind_level{}'.format(level)] - features_df['first_mean-vw_level{}'.format(level)]
        for level_1,level_2 in [(1000,700),(100,50),(1000,50),(1000,300),(70,50),(150,50)]:   
            features_df['uwind_level_{}_{}_diff'.format(level_1,level_2)] = features_df['uwind_level{}'.format(level_1)] -  features_df['uwind_level{}'.format(level_2)]
            features_df['p14d-uw_level_{}_{}_diff'.format(level_1,level_2)] = features_df['p14d-uw_level{}'.format(level_1)] -  features_df['p14d-uw_level{}'.format(level_2)]
            features_df['first_mean-uw_level_{}_{}_diff'.format(level_1,level_2)] = features_df['first_mean-uw_level{}'.format(level_1)] -  features_df['first_mean-uw_level{}'.format(level_2)]
            features_df['first_mean-vw_level_{}_{}_diff'.format(level_1,level_2)] = features_df['first_mean-vw_level{}'.format(level_1)] -  features_df['first_mean-vw_level{}'.format(level_2)]
            features_df['first-uw_level_{}_{}_diff'.format(level_1,level_2)] = features_df['first_uwind_level{}'.format(level_1)] -  features_df['first_uwind_level{}'.format(level_2)]
            features_df['first-vw_level_{}_{}_diff'.format(level_1,level_2)] = features_df['first_vwind_level{}'.format(level_1)] -  features_df['first_vwind_level{}'.format(level_2)]
            #features_df['first_mean-wind_module_level_{}_{}_diff'.format(level_1,level_2)] = features_df['first_wind_module_level{}'.format(level_1)] -  features_df['first_wind_module_level{}'.format(level_2)]
        
        return features_df.drop(['lat','lon','lat_near_2','lon_near_2','lat_near_3','lon_near_3','lat_near_4','lon_near_4','time',
                                'bearing_from_nearest','latitude','longitude',
                                'uwind_near_2_level_{}'.format(level),'uwind_near_3_level_{}'.format(level),
                                'uwind_near_4_level_{}'.format(level),'vwind_near_2_level_{}'.format(level),
                                'vwind_near_3_level_{}'.format(level),'vwind_near_4_level_{}'.format(level),
                                'first_uwind_near_2_level_{}'.format(level),'first_uwind_near_3_level_{}'.format(level),
                                'first_uwind_near_4_level_{}'.format(level),'first_vwind_near_2_level_{}'.format(level),
                                'first_vwind_near_3_level_{}'.format(level),'first_vwind_near_4_level_{}'.format(level)],
                                axis=1,errors='ignore')
    
    def make_winds_features(self,df):
        start = time.time()
        #print('Reading winds dataset...')
        self.read_winds_set()
        #print("Reading dataset done! Time {:.2f}".format(time.time()-start))
        start = time.time()
        #print('Making wind features...')
        winds_features = self.extract_winds_features(df)
        #print('Making wind features done! Time {:.2f}'.format(time.time()-start))
        return pd.DataFrame(winds_features).set_index('fire_id')
    
    def read_winds_set(self):       
        ncep_data = []
        for var in ['uwnd','vwnd']:
            #print(var)
            year = 2019
            dataset_filename = '{}/noaa/ncep/{}.{}.nc'.format(DATASETS_PATH,var, year)
            #dataset_filename = '{}.{}.nc'.format(var, year)
            try:
                array = xarray.open_dataset(dataset_filename)
            except:
                dataset_filename = '{}.{}.nc'.format(var, year)
                array = xarray.open_dataset(dataset_filename)
            array = array.where((array.lat >= self.lat_min) &
                  (array.lat <= self.lat_max) & 
                  (array.lon <= self.lon_max) & 
                  (array.lon >= self.lon_min) & 
                  (array.level.isin(self.base_levels)),drop=True)
            ncep_data.append(array)
            my_max_date = datetime.date(2019,11,13)
            max_date = datetime.datetime.utcfromtimestamp(int((array.time.max().item(0)/10e8)+3600*24)).date()
            print('{} {}'.format(var,max_date))
            if max_date < my_max_date:
                dataset_filename = '{}.{}.nc'.format(var, year)
                array = xarray.open_dataset(dataset_filename)
                array = array.where((array.lat >= self.lat_min) &
                  (array.lat <= self.lat_max) & 
                  (array.lon <= self.lon_max) & 
                  (array.lon >= self.lon_min) & 
                  (array.level.isin(self.base_levels)),drop=True)
                array = array.sel(time=slice(max_date.strftime('%Y-%m-%d'),'2019-11-13'))
                ncep_data.append(array)
        full_array = xarray.merge(ncep_data)
        self.ncep_data = full_array
        
    def make_transform_features(self,X):
        with io.open('transform_dict-2.json','r',encoding='utf-8') as f:
            transform_dict = json.load(f)
        
        X['lat-lon'] = X['lat'].astype(str) + '-' + X['lon'].astype(str)
        var_dict = {'t':'temperature','h':'humidity','uw':'uwind'}
        for level in self.base_levels:
            for var in ['t','h','uw']:
                new_var = var_dict[var]
                for col in ['p14d_diff-{}_level{}_latlon'.format(var,level),
                  'diff_first_mean-{}_level{}_latlon'.format(var,level),
                  '{}_level{}_latlon'.format(new_var,level)]:
                    if col in transform_dict.keys():
                        X[col] = X['lat-lon'].map(transform_dict[col])
                        bad_ind = X[X[col].isnull()].index
                        print(col,len(bad_ind))
                        if len(bad_ind) > 0:
                            if col == 'p14d_diff-{}_level{}_latlon'.format(var,level):
                                X.loc[bad_ind,col] = X.loc[bad_ind].groupby('lat-lon')['p14d_diff-{}_level{}'.format(var,level)].transform('mean')
                            elif col == 'diff_first_mean-{}_level{}_latlon'.format(var,level):
                                X.loc[bad_ind,col] = X.loc[bad_ind].groupby('lat-lon')['diff_first_mean-{}_level{}'.format(var,level)].transform('mean')
                            elif col == '{}_level{}_latlon'.format(new_var,level):
                                X.loc[bad_ind,col] = X.loc[bad_ind].groupby('lat-lon')['{}_level{}'.format(new_var,level)].transform('mean')
                        
        return X.drop('lat-lon',axis=1,errors='ignore')
        
    def make_city_features(self,X):
        biggest_cities = pd.read_csv('biggest_cities.csv')
        biggest_cities.loc[923,'Население'] = 400
        biggest_cities['Население'] = biggest_cities['Население'].astype(np.int)
        for _ind in [506,782,863]:
            biggest_cities.loc[_ind,'Город'] = biggest_cities.loc[_ind,'Регион']
        biggest_cities.loc[1066,'Город'] = 'Урус-Мартан'
        biggest_cities = biggest_cities[['Широта','Долгота','Население','Город']]

        kd_tree_cities = KDTree(biggest_cities[['Долгота','Широта']])
        X['city_lon'] = biggest_cities.iloc[kd_tree_cities.query(X[['longitude','latitude']],1,return_distance=False)[:,0]]['Долгота'].values
        X['city_lat'] = biggest_cities.iloc[kd_tree_cities.query(X[['longitude','latitude']],1,return_distance=False)[:,0]]['Широта'].values
        X['city_population'] = biggest_cities.iloc[kd_tree_cities.query(X[['longitude','latitude']],1,return_distance=False)[:,0]]['Население'].values
        X['distance_from_city'] = X[['city_lon','city_lat','latitude','longitude']]\
                                   .apply(lambda x: calc_distance(x['city_lon'],x['city_lat'],
                                                        x['longitude'],x['latitude']),axis=1)
                                                        
        for _file in ['field_coords','forest_coords','nature_forests']:
            forest_df = pd.read_csv('{}.csv'.format(_file))
            kd_tree_nature = KDTree(forest_df[['longitude','latitude']])
            X['{}_lon'.format(_file)] = forest_df.iloc[kd_tree_nature.query(X[['longitude','latitude']],1,return_distance=False)[:,0]]['longitude'].values
            X['{}_lat'.format(_file)] = forest_df.iloc[kd_tree_nature.query(X[['longitude','latitude']],1,return_distance=False)[:,0]]['latitude'].values
            X['distance_from_{}'.format(_file)] = X[['{}_lon'.format(_file),'{}_lat'.format(_file),'latitude','longitude']]\
                                   .apply(lambda x: calc_distance(x['{}_lon'.format(_file)],x['{}_lat'.format(_file)],
                                                   x['longitude'],x['latitude']),axis=1)
        return X
        
    def make_all_features(self,df,full_train):
        self.full_train = full_train
        if df['date'].dtype== 'O':
            df['date'] = pd.to_datetime(df['date'])
            
        df['lon'] = fm.lat_lons[fm.kd_tree.query(df[['longitude','latitude']],1,return_distance=False)[:,0]][:,0]
        df['lat'] = fm.lat_lons[fm.kd_tree.query(df[['longitude','latitude']],1,return_distance=False)[:,0]][:,1]
        df['time'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        
        for nearest in [2,3,4]:
            df['lon_near_{}'.format(nearest)] = fm.lat_lons[fm.kd_tree.query(df[['longitude','latitude']],4,return_distance=False)[:,nearest-1]][:,0]
            df['lat_near_{}'.format(nearest)] = fm.lat_lons[fm.kd_tree.query(df[['longitude','latitude']],4,return_distance=False)[:,nearest-1]][:,1]  
            
        features = self.make_raw_features(df)
        features = features.merge(self.make_spatial_features(df),left_index=True,right_index=True)
        features = features.merge(self.make_temp_rhum_features(df),left_index=True,right_index=True)
        features = features.merge(self.make_winds_features(df),left_index=True,right_index=True)
        #self.divide_count_features(features)
        features = self.make_transform_features(features)
        features = self.make_city_features(features)
                    
        return features
    
    def make_labels(self,df):
        return df.set_index('fire_id')['fire_type']
    
    def make_train_set(self,df):
        X = self.make_all_features(df)
        y = self.make_labels(df)
        return X,y
        
        
if __name__ == '__main__':
    start = time.time()
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    df_points = pd.read_csv(input_csv)
    #wildfires_check = pd.read_csv('wildfires_check.csv')
    #df_train = pd.concat([pd.read_csv('wildfires_train.csv'),
    #                  wildfires_check],ignore_index=True)
    df_train = pd.read_csv('wildfires_train.csv')
                      
    #if len(df_points) != len(wildfires_check) and len(set(df_points['fire_id'].unique()).intersection(set(wildfires_check['fire_id']))) < len(wildfires_check['fire_id'].unique()):
    #    df_train = pd.concat([df_train,df_points],ignore_index=True)
    print(df_train.shape)

    fm = FeaturesMaker()
    X = fm.make_all_features(df_points,df_train)

    lgb_model = lgb.Booster(model_file='lgbm_model_3200_iter_214_filtered_features-without_month.dmp')

    X.replace({np.inf:np.nan,-np.inf:np.nan},inplace=True)
    X.fillna(method='ffill',inplace=True)
    X.fillna(method='bfill',inplace=True)
    print(X[model_features].shape)
    predictions = lgb_model.predict(X[model_features])
    X.to_csv('test-features-check.csv')
    
    df_predictions = pd.DataFrame(
        predictions,
        index=X.index,
        columns=[
            'fire_{}_prob'.format(class_id)
            for class_id in range(1,12,1)
        ],
    )

    df_predictions.to_csv(output_csv)
