//
// Created by lenovo on 2025/5/8.
//

#ifndef AIDATA_H
#define AIDATA_H

#define MAX_LINE_LEN 1024
#define MAX_SAMPLES 506
#define NUM_FEATURES 13
#define NUM_SELECTED_FEATURES 4

// 数据结构定义
typedef struct {
    double features[NUM_FEATURES];  // 13个特征
    double target;                  // 房价MEDV
} Sample;

// 存储归一化参数
typedef struct {
    double mean;
    double std;
} NormParam;

// 函数声明
void load_data(const char* filename, Sample* samples, int* num_samples);
void normalize_features(Sample* samples, int num_samples, NormParam* norm_params);
void calculate_correlation(const Sample* samples, int num_samples, double* correlation);
void select_features(const double* correlation, int* selected_indices);
void train_model(const Sample* samples, int num_samples, const int* selected_indices, double* weights, double* bias);
double predict(const double* features, const int* selected_indices, const double* weights, double bias);
double calculate_rmse(const Sample* samples, int num_samples, const int* selected_indices, const double* weights, double bias);
void denormalize_features(Sample* samples, int num_samples, const NormParam* norm_params);

#endif //AIDATA_H
