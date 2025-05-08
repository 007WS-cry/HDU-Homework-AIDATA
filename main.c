#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <locale.h>
#include "aidata.h"

int main() {
    // 设置宽字符本地化环境
    setlocale(LC_ALL, "");

    Sample samples[MAX_SAMPLES];
    int num_samples = 0;
    NormParam norm_params[NUM_FEATURES + 1]; // +1 for target
    double correlation[NUM_FEATURES];
    int selected_indices[NUM_SELECTED_FEATURES];
    double weights[NUM_SELECTED_FEATURES];
    double bias;

    // 1. 加载数据
    wprintf(L"正在加载数据...\n");
    load_data("housing.txt", samples, &num_samples);
    wprintf(L"成功加载 %d 个样本\n", num_samples);

    // 2. 数据归一化处理
    wprintf(L"正在归一化数据...\n");
    normalize_features(samples, num_samples, norm_params);

    // 3. 计算相关系数
    wprintf(L"正在计算相关系数...\n");
    calculate_correlation(samples, num_samples, correlation);

    // 输出相关系数
    wprintf(L"\n各特征与房价(MEDV)的相关系数:\n");
    char* feature_names[NUM_FEATURES] = {
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    };

    for (int i = 0; i < NUM_FEATURES; i++) {
        wprintf(L"%-8hs: %f\n", feature_names[i], correlation[i]);
    }

    // 4. 特征选择 - 挑选最相关的4个特征
    wprintf(L"\n选择最相关的4个特征...\n");
    select_features(correlation, selected_indices);

    wprintf(L"选择的特征索引: ");
    for (int i = 0; i < NUM_SELECTED_FEATURES; i++) {
        wprintf(L"%hs ", feature_names[selected_indices[i]]);
        if (i < NUM_SELECTED_FEATURES - 1) {
            wprintf(L", ");
        }
    }
    wprintf(L"\n");

    // 5. 训练多元线性回归模型
    wprintf(L"\n训练多元线性回归模型...\n");
    train_model(samples, num_samples, selected_indices, weights, &bias);

    // 输出模型参数
    wprintf(L"模型参数:\n");
    wprintf(L"偏置项 (bias): %f\n", bias);
    for (int i = 0; i < NUM_SELECTED_FEATURES; i++) {
        wprintf(L"权重 %hs: %f\n", feature_names[selected_indices[i]], weights[i]);
    }

    // 6. 评估模型
    double rmse = calculate_rmse(samples, num_samples, selected_indices, weights, bias);
    wprintf(L"\n模型评估:\n");
    wprintf(L"均方根误差 (RMSE): %f\n", rmse);

    // 7. 还原数据 (反归一化)
    denormalize_features(samples, num_samples, norm_params);

    // 8. 使用原始尺度的模型进行预测示例
    wprintf(L"\n预测示例 (使用前5个样本):\n");
    wprintf(L"%-8ls %-8ls %-8ls\n", L"实际值", L"预测值", L"误差");

    for (int i = 0; i < 5 && i < num_samples; i++) {
        double features_to_predict[NUM_FEATURES];
        for (int j = 0; j < NUM_FEATURES; j++) {
            features_to_predict[j] = samples[i].features[j];
        }

        // 预测时需要先归一化
        for (int j = 0; j < NUM_FEATURES; j++) {
            features_to_predict[j] = (features_to_predict[j] - norm_params[j].mean) / norm_params[j].std;
        }

        double predicted = predict(features_to_predict, selected_indices, weights, bias);
        // 反归一化预测结果
        predicted = predicted * norm_params[NUM_FEATURES].std + norm_params[NUM_FEATURES].mean;

        wprintf(L"%-8.2f %-8.2f %-8.2f\n",
               samples[i].target,
               predicted,
               fabs(samples[i].target - predicted));
    }
    system("pause");

    return 0;
}