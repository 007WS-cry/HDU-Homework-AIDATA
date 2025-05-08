//
// Created by lenovo on 2025/5/8.
//

#include "aidata.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 加载数据
void load_data(const char* filename, Sample* samples, int* num_samples) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        wprintf(L"无法打开文件: %hs\n", filename);
        exit(1);
    }

    char line[MAX_LINE_LEN];
    // 跳过标题行
    fgets(line, MAX_LINE_LEN, file);

    *num_samples = 0;
    while (fgets(line, MAX_LINE_LEN, file) && *num_samples < MAX_SAMPLES) {
        char* token = strtok(line, " ");
        int i = 0;

        while (token != NULL && i <= NUM_FEATURES) {
            if (i < NUM_FEATURES) {
                samples[*num_samples].features[i] = atof(token);
            } else {
                samples[*num_samples].target = atof(token);
            }
            token = strtok(NULL, " ");
            i++;
        }

        (*num_samples)++;
    }

    fclose(file);
}

// 归一化特征
void normalize_features(Sample* samples, int num_samples, NormParam* norm_params) {
    // 计算均值和标准差
    for (int j = 0; j < NUM_FEATURES; j++) {
        double sum = 0, sum_sq = 0;

        for (int i = 0; i < num_samples; i++) {
            sum += samples[i].features[j];
            sum_sq += samples[i].features[j] * samples[i].features[j];
        }

        double mean = sum / num_samples;
        double variance = sum_sq / num_samples - mean * mean;
        double std = sqrt(variance);

        // 存储归一化参数
        norm_params[j].mean = mean;
        norm_params[j].std = std;

        // 应用归一化
        for (int i = 0; i < num_samples; i++) {
            samples[i].features[j] = (samples[i].features[j] - mean) / std;
        }
    }

    // 对目标变量也进行归一化
    double sum = 0, sum_sq = 0;
    for (int i = 0; i < num_samples; i++) {
        sum += samples[i].target;
        sum_sq += samples[i].target * samples[i].target;
    }

    double mean = sum / num_samples;
    double variance = sum_sq / num_samples - mean * mean;
    double std = sqrt(variance);

    norm_params[NUM_FEATURES].mean = mean;
    norm_params[NUM_FEATURES].std = std;

    for (int i = 0; i < num_samples; i++) {
        samples[i].target = (samples[i].target - mean) / std;
    }
}

// 计算相关系数
void calculate_correlation(const Sample* samples, int num_samples, double* correlation) {
    for (int j = 0; j < NUM_FEATURES; j++) {
        double sum_x = 0, sum_y = 0;
        double sum_xx = 0, sum_yy = 0, sum_xy = 0;

        for (int i = 0; i < num_samples; i++) {
            const double x = samples[i].features[j];
            const double y = samples[i].target;

            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
        }

        // 计算皮尔逊相关系数
        double numerator = num_samples * sum_xy - sum_x * sum_y;
        double denominator = sqrt((num_samples * sum_xx - sum_x * sum_x) * (num_samples * sum_yy - sum_y * sum_y));

        correlation[j] = numerator / denominator;
    }
}

// 选择最相关的特征
void select_features(const double* correlation, int* selected_indices) {
    // 为了保留原始相关系数，创建一个副本
    double corr_copy[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {
        corr_copy[i] = fabs(correlation[i]);  // 使用绝对值，因为负相关也是强相关
    }

    // 选择最相关的4个特征
    for (int k = 0; k < NUM_SELECTED_FEATURES; k++) {
        double max_corr = -1;
        int max_idx = -1;

        for (int i = 0; i < NUM_FEATURES; i++) {
            if (corr_copy[i] > max_corr) {
                max_corr = corr_copy[i];
                max_idx = i;
            }
        }

        selected_indices[k] = max_idx;
        corr_copy[max_idx] = -1;  // 标记该特征已选择
    }
}

// 训练多元线性回归模型 (使用正规方程)
void train_model(const Sample* samples, int num_samples, const int* selected_indices, double* weights, double* bias) {
    // 初始化矩阵求解相关变量
    double X_T_X[NUM_SELECTED_FEATURES][NUM_SELECTED_FEATURES] = {0};
    double X_T_y[NUM_SELECTED_FEATURES] = {0};

    // 计算 X^T * X 和 X^T * y
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < NUM_SELECTED_FEATURES; j++) {
            for (int k = 0; k < NUM_SELECTED_FEATURES; k++) {
                X_T_X[j][k] += samples[i].features[selected_indices[j]] * samples[i].features[selected_indices[k]];
            }
            X_T_y[j] += samples[i].features[selected_indices[j]] * samples[i].target;
        }
    }

    // 高斯消元法求解线性方程组
    // 前向消元
    for (int i = 0; i < NUM_SELECTED_FEATURES; i++) {
        // 对角元素归一化
        double pivot = X_T_X[i][i];
        for (int j = i; j < NUM_SELECTED_FEATURES; j++) {
            X_T_X[i][j] /= pivot;
        }
        X_T_y[i] /= pivot;

        // 消元
        for (int k = i + 1; k < NUM_SELECTED_FEATURES; k++) {
            double factor = X_T_X[k][i];
            for (int j = i; j < NUM_SELECTED_FEATURES; j++) {
                X_T_X[k][j] -= factor * X_T_X[i][j];
            }
            X_T_y[k] -= factor * X_T_y[i];
        }
    }

    // 回代求解
    for (int i = NUM_SELECTED_FEATURES - 1; i >= 0; i--) {
        weights[i] = X_T_y[i];
        for (int j = i + 1; j < NUM_SELECTED_FEATURES; j++) {
            weights[i] -= X_T_X[i][j] * weights[j];
        }
    }

    // 计算偏置项
    double y_mean = 0, X_mean[NUM_SELECTED_FEATURES] = {0};
    for (int i = 0; i < num_samples; i++) {
        y_mean += samples[i].target;
        for (int j = 0; j < NUM_SELECTED_FEATURES; j++) {
            X_mean[j] += samples[i].features[selected_indices[j]];
        }
    }
    y_mean /= num_samples;
    for (int j = 0; j < NUM_SELECTED_FEATURES; j++) {
        X_mean[j] /= num_samples;
    }

    *bias = y_mean;
    for (int j = 0; j < NUM_SELECTED_FEATURES; j++) {
        *bias -= weights[j] * X_mean[j];
    }
}

// 预测函数
double predict(const double* features, const int* selected_indices, const double* weights, double bias) {
    double result = bias;
    for (int j = 0; j < NUM_SELECTED_FEATURES; j++) {
        result += weights[j] * features[selected_indices[j]];
    }
    return result;
}

// 计算均方根误差 (RMSE)
double calculate_rmse(const Sample* samples, int num_samples, const int* selected_indices, const double* weights, double bias) {
    double sum_squared_error = 0;

    for (int i = 0; i < num_samples; i++) {
        double predicted = predict(samples[i].features, selected_indices, weights, bias);
        double error = predicted - samples[i].target;
        sum_squared_error += error * error;
    }

    return sqrt(sum_squared_error / num_samples);
}

// 反归一化数据
void denormalize_features(Sample* samples, int num_samples, const NormParam* norm_params) {
    // 反归一化特征
    for (int j = 0; j < NUM_FEATURES; j++) {
        for (int i = 0; i < num_samples; i++) {
            samples[i].features[j] = samples[i].features[j] * norm_params[j].std + norm_params[j].mean;
        }
    }

    // 反归一化目标变量
    for (int i = 0; i < num_samples; i++) {
        samples[i].target = samples[i].target * norm_params[NUM_FEATURES].std + norm_params[NUM_FEATURES].mean;
    }
}