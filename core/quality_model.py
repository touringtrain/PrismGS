import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

class GaussianCompressionOptimizer:
    def __init__(self):
        # 初始化质量数据
        qp_levels = np.array([0, 10, 20, 30, 40, 50])
        self.quality_data = {
            'position_low': np.array([34.7470, 34.7466, 34.7456, 34.7352, 34.6413, 33.8800]),
            'color': np.array([35.2377, 35.1931, 34.7456, 31.3446, 23.8429, 21.7427]),
            'opacity': np.array([34.7560, 34.7555, 34.7456, 34.7009, 33.9201, 27.2833]),
            'scale': np.array([34.9311, 34.9214, 34.7456, 32.3086, 19.3342, 19.3121]),
            'rotation': np.array([35.4619, 35.3728, 34.7456, 30.6098, 22.6330, 23.1701])
        }
        
        # 新增：全属性统一QP的质量数据 (除了position_high固定QP=0)
        self.unified_qp_quality = np.array([
            36.2693, 36.2334, 36.0995, 35.7345, 34.7456,
            32.3418, 27.4024, 22.3123, 17.4712, 14.8614, 14.8614
        ])
        self.unified_qp_levels = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        
        # 码率数据
        self.bitrate_data = {
            'position_high': {0: 49.8, 10: 49.8, 20: 49.8, 30: 49.8, 40: 49.8, 50: 49.8},
            'position_low': {0: 167.9, 10: 140.0, 20: 96.0, 30: 64.0, 40: 38.0, 50: 8.0},
            'SH_color': {0: 36.0, 10: 30.0, 20: 14.0, 30: 8.0, 40: 3.0, 50: 1.0},
            'opacity': {0: 54.0, 10: 42.0, 20: 19.0, 30: 8.0, 40: 2.4, 50: 1.0},
            'scale': {0: 59.0, 10: 48.0, 20: 21.0, 30: 11.5, 40: 7.0, 50: 1.0},
            'rotation': {0: 55.0, 10: 38.0, 20: 18.0, 30: 9.0, 40: 4.0, 50: 1.0}
        }
        
        self.channel_multipliers = {
            'position_high': 3, 'position_low': 3, 'SH_color': 3, 
            'opacity': 1, 'scale': 3, 'rotation': 4
        }
        
        self.config_mapping = {
            'position_low': 'position_low', 'color': 'SH_color', 
            'opacity': 'opacity', 'scale': 'scale', 'rotation': 'rotation'
        }
        
        # 构建插值函数
        self.quality_functions = {}
        self.bitrate_functions = {}
        
        # 使用统一QP数据校准质量预测模型
        self._calibrate_with_unified_data()
        
        for attr, bitrates in self.bitrate_data.items():
            bitrate_points = np.array([bitrates[qp] for qp in qp_levels])
            self.bitrate_functions[attr] = interp1d(qp_levels, bitrate_points, 
                                                   kind='linear', fill_value="extrapolate")
        
        self.base_psnr = 34.7456  # 基准PSNR

    def _calibrate_with_unified_data(self):
        """使用统一QP实验数据校准质量预测模型"""
        
        # 创建统一QP数据的插值函数
        self.unified_quality_func = CubicSpline(self.unified_qp_levels, self.unified_qp_quality)
        
        # 计算控制变量实验与统一QP实验之间的系统误差
        control_qp_points = np.array([0, 10, 20, 30, 40, 50])
        calibration_factors = {}
        
        for attr, control_qualities in self.quality_data.items():
            # 在相同QP点上比较两种实验
            unified_qualities_at_control_points = self.unified_quality_func(control_qp_points)
            
            # 计算校准因子（考虑系统偏差）
            ratios = unified_qualities_at_control_points / control_qualities
            calibration_factors[attr] = np.mean(ratios)
            
            print(f"{attr} 校准因子: {calibration_factors[attr]:.4f}")
        
        # 应用校准因子并创建质量函数
        qp_levels = np.array([0, 10, 20, 30, 40, 50])
        for attr, control_qualities in self.quality_data.items():
            # 应用校准
            calibrated_qualities = control_qualities * calibration_factors[attr]
            
            # 归一化到QP=20
            base_quality = calibrated_qualities[2]  # QP=20
            normalized_qualities = calibrated_qualities / base_quality
            
            # 创建插值函数
            self.quality_functions[attr] = CubicSpline(qp_levels, normalized_qualities)
        
        # 更新基准PSNR
        self.base_psnr = self.unified_quality_func(20)
        print(f"基准PSNR更新为: {self.base_psnr:.4f} dB (来自统一QP实验)")

    def predict_quality_basic(self, config):
        """基础质量预测（使用校准后的控制变量数据）"""
        total_quality = 1.0
        for attr, qp in config.items():
            quality_contribution = self.quality_functions[attr](qp)
            total_quality *= quality_contribution
        return total_quality * self.base_psnr

    def predict_quality_enhanced(self, config):
        """
        增强质量预测 - 结合统一QP数据进行修正
        
        使用混合方法：
        1. 基础预测：基于可分离假设
        2. 统一QP参考：基于平均QP的统一实验数据
        3. 加权融合：根据配置的均匀程度调整权重
        """
        # 基础预测
        base_prediction = self.predict_quality_basic(config)
        
        # 计算配置的平均QP和均匀度
        qp_values = list(config.values())
        avg_qp = np.mean(qp_values)
        qp_std = np.std(qp_values)  # QP值的标准差，衡量均匀度
        
        # 从统一QP实验获取参考质量
        unified_reference = self.unified_quality_func(avg_qp)
        
        # 计算权重：配置越均匀，越相信统一QP数据
        # 当所有QP相同时(qp_std=0)，完全相信统一QP数据
        # 当QP差异很大时(qp_std很大)，更相信基础预测
        uniformity_weight = np.exp(-qp_std / 10.0)  # 指数衰减，标准差10时权重约0.37
        
        # 混合预测
        mixed_prediction = (uniformity_weight * unified_reference + 
                           (1 - uniformity_weight) * base_prediction)
        
        return mixed_prediction

    def predict_quality_correction_model(self, config):
        """
        基于校正模型的质量预测
        
        使用统一QP数据训练一个校正模型来修正基础预测
        """
        # 基础预测
        base_pred = self.predict_quality_basic(config)
        
        # 提取特征用于校正
        qp_values = list(config.values())
        avg_qp = np.mean(qp_values)
        qp_std = np.std(qp_values)
        max_qp = max(qp_values)
        min_qp = min(qp_values)
        qp_range = max_qp - min_qp
        
        # 从统一QP实验获取参考
        unified_ref = self.unified_quality_func(avg_qp)
        
        # 计算校正因子（基于经验模型）
        # 当配置均匀时，校正因子接近1（基础预测准确）
        # 当配置不均匀时，需要更大的校正
        if qp_std < 5:  # 相对均匀的配置
            correction_factor = 1.0 + (unified_ref - base_pred) / (2 * base_pred)
        else:  # 不均匀配置
            # 考虑QP差异的影响
            qp_imbalance_penalty = 0.02 * qp_std  # 标准差每增加1，质量下降0.02dB
            correction_factor = (unified_ref - qp_imbalance_penalty) / base_pred
        
        # 应用校正，但限制校正幅度
        correction_factor = np.clip(correction_factor, 0.8, 1.2)
        corrected_prediction = base_pred * correction_factor
        
        return corrected_prediction

    def predict_quality(self, config, method='enhanced'):
        """
        统一的质量预测接口
        
        Args:
            config: 配置字典
            method: 预测方法 ('basic', 'enhanced', 'correction')
        """
        if method == 'basic':
            return self.predict_quality_basic(config)
        elif method == 'enhanced':
            return self.predict_quality_enhanced(config)
        elif method == 'correction':
            return self.predict_quality_correction_model(config)
        else:
            raise ValueError("方法必须是 'basic', 'enhanced', 或 'correction'")

    def validate_prediction_accuracy(self):
        """验证不同预测方法的准确性"""
        print("\n" + "="*50)
        print("预测准确性验证")
        print("="*50)
        
        # 测试配置
        test_configs = [
            # 统一QP配置（可以直接验证）
            {attr: 0 for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation']},
            {attr: 10 for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation']},
            {attr: 20 for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation']},
            {attr: 30 for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation']},
            {attr: 40 for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation']},
            {attr: 45 for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation']},
            
            # 非均匀配置
            {'position_low': 5, 'color': 15, 'opacity': 10, 'scale': 20, 'rotation': 25},
            {'position_low': 35, 'color': 25, 'opacity': 40, 'scale': 30, 'rotation': 20},
            {'position_low': 10, 'color': 40, 'opacity': 20, 'scale': 15, 'rotation': 35},
        ]
        
        methods = ['basic', 'enhanced', 'correction']
        errors = {method: [] for method in methods}
        
        for i, config in enumerate(test_configs):
            qp_values = list(config.values())
            avg_qp = np.mean(qp_values)
            qp_std = np.std(qp_values)
            
            # 对于统一QP配置，我们可以与真实值比较
            if len(set(qp_values)) == 1:  # 所有QP相同
                actual_quality = self.unified_quality_func(avg_qp)
                has_ground_truth = True
            else:
                actual_quality = None
                has_ground_truth = False
            
            print(f"\n配置 {i+1}: {config}")
            print(f"  平均QP: {avg_qp:.1f}, QP标准差: {qp_std:.1f}")
            
            for method in methods:
                predicted = self.predict_quality(config, method)
                
                if has_ground_truth:
                    error = abs(predicted - actual_quality)
                    errors[method].append(error)
                    print(f"  {method:>10}: {predicted:6.2f} dB (误差: {error:5.2f} dB)")
                else:
                    print(f"  {method:>10}: {predicted:6.2f} dB")
            
            if has_ground_truth:
                print(f"  真实值: {actual_quality:6.2f} dB")
        
        # 统计误差
        print("\n" + "="*50)
        print("误差统计 (仅统一QP配置):")
        for method in methods:
            if errors[method]:
                mean_error = np.mean(errors[method])
                max_error = np.max(errors[method])
                print(f"  {method:>10}: 平均误差 {mean_error:.3f} dB, 最大误差 {max_error:.3f} dB")

    def calculate_bitrate(self, config):
        """计算配置的总码率"""
        total_bitrate = 49.8 * self.channel_multipliers['position_high']  # position_high固定
        
        for config_key, data_key in self.config_mapping.items():
            qp_value = config[config_key]
            attribute_bitrate = self.bitrate_functions[data_key](qp_value) * self.channel_multipliers[data_key]
            total_bitrate += attribute_bitrate
        
        return total_bitrate

    # 保留原有的优化方法，但使用增强的质量预测
    def optimize_single_target(self, target_bitrate, prediction_method='enhanced'):
        """针对单个目标码率进行优化"""
        def objective(qp_vector):
            config = {
                'position_low': qp_vector[0], 'color': qp_vector[1],
                'opacity': qp_vector[2], 'scale': qp_vector[3], 'rotation': qp_vector[4]
            }
            return -self.predict_quality(config, prediction_method)

        def constraint(qp_vector):
            config = {
                'position_low': qp_vector[0], 'color': qp_vector[1],
                'opacity': qp_vector[2], 'scale': qp_vector[3], 'rotation': qp_vector[4]
            }
            return target_bitrate - self.calculate_bitrate(config)

        initial_guess = [20, 20, 20, 20, 20]
        bounds = [(0, 50)] * 5
        constraints = {'type': 'ineq', 'fun': constraint}
        
        result = minimize(objective, initial_guess, bounds=bounds, 
                         constraints=constraints, method='SLSQP')
        
        if result.success:
            continuous_config = dict(zip(['position_low', 'color', 'opacity', 'scale', 'rotation'], 
                                       result.x))
            # 现在支持5为步长的QP值
            discrete_config = {k: int(round(v / 5) * 5) for k, v in continuous_config.items()}
            discrete_config = {k: min(50, max(0, v)) for k, v in discrete_config.items()}
            
            if self.calculate_bitrate(discrete_config) <= target_bitrate:
                return discrete_config
        return None

    def generate_pareto_frontier(self, bitrate_range=None, num_points=20, method='smart_sampling'):
        """
        生成Pareto前沿
        
        Args:
            bitrate_range: (min_bitrate, max_bitrate) 码率范围
            num_points: Pareto前沿上的点数
            method: 生成方法 ('uniform', 'smart_sampling', 'dp')
        
        Returns:
            pareto_front: Pareto前沿上的配置列表，每个元素为 (config, quality, bitrate)
        """
        
        if bitrate_range is None:
            # 自动确定码率范围
            min_config = {attr: 50 for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation']}
            max_config = {attr: 0 for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation']}
            bitrate_range = (self.calculate_bitrate(min_config), self.calculate_bitrate(max_config))
        
        min_bitrate, max_bitrate = bitrate_range
        
        if method == 'uniform':
            return self._generate_pareto_uniform(min_bitrate, max_bitrate, num_points)
        elif method == 'smart_sampling':
            return self._generate_pareto_smart_sampling(min_bitrate, max_bitrate, num_points)
        elif method == 'dp':
            return self._generate_pareto_dp(min_bitrate, max_bitrate, num_points)
        else:
            raise ValueError("方法必须是 'uniform', 'smart_sampling', 或 'dp'")

    def _generate_pareto_uniform(self, min_bitrate, max_bitrate, num_points):
        """均匀采样方法生成Pareto前沿"""
        pareto_front = []
        
        # 在码率范围内均匀采样
        target_bitrates = np.linspace(min_bitrate, max_bitrate, num_points)
        
        for target_bitrate in target_bitrates:
            config = self.optimize_single_target(target_bitrate)
            if config:
                quality = self.predict_quality(config)
                bitrate = self.calculate_bitrate(config)
                pareto_front.append((config, quality, bitrate))
        
        # 过滤非支配解
        return self._filter_pareto_front(pareto_front)

    def _generate_pareto_smart_sampling(self, min_bitrate, max_bitrate, num_points):
        """智能采样方法生成Pareto前沿"""
        # 首先在关键点采样
        key_bitrates = [
            min_bitrate, max_bitrate,
            (min_bitrate + max_bitrate) / 2,
            min_bitrate + (max_bitrate - min_bitrate) * 0.25,
            min_bitrate + (max_bitrate - min_bitrate) * 0.75
        ]
        
        pareto_front = []
        for bitrate in key_bitrates:
            config = self.optimize_single_target(bitrate)
            if config:
                quality = self.predict_quality(config)
                bitrate_actual = self.calculate_bitrate(config)
                pareto_front.append((config, quality, bitrate_actual))
        
        # 在质量变化剧烈的区域增加采样密度
        pareto_front.sort(key=lambda x: x[2])  # 按码率排序
        
        # 迭代细化
        for _ in range(3):
            new_points = []
            for i in range(len(pareto_front) - 1):
                current_config, current_quality, current_bitrate = pareto_front[i]
                next_config, next_quality, next_bitrate = pareto_front[i + 1]
                
                # 如果两个点之间的质量下降很大，在中间增加采样
                quality_drop = current_quality - next_quality
                if quality_drop > 2.0:  # 质量下降超过2dB
                    mid_bitrate = (current_bitrate + next_bitrate) / 2
                    mid_config = self.optimize_single_target(mid_bitrate)
                    if mid_config:
                        mid_quality = self.predict_quality(mid_config)
                        mid_bitrate_actual = self.calculate_bitrate(mid_config)
                        new_points.append((mid_config, mid_quality, mid_bitrate_actual))
            
            pareto_front.extend(new_points)
            pareto_front = self._filter_pareto_front(pareto_front)
            pareto_front.sort(key=lambda x: x[2])
        
        return pareto_front

    def _generate_pareto_dp(self, min_bitrate, max_bitrate, num_points):
        """动态规划方法生成Pareto前沿，并使用enhanced预测进行后处理校正"""
        attributes = ['position_low', 'color', 'opacity', 'scale', 'rotation']
        qp_options = list(range(0, 51, 1))  # 0-50每10为步长
        
        # 离散化码率空间
        bitrate_step = 10  # KB
        bitrate_bins = range(int(min_bitrate), int(max_bitrate) + 1, bitrate_step)
        
        # 第一阶段：使用基础预测进行DP（保持最优子结构）
        dp = [{} for _ in range(len(attributes) + 1)]
        dp[0][0] = (1.0, {})  # 初始状态
        
        for i, attr in enumerate(attributes):
            current_stage = i + 1
            dp[current_stage] = {}
            
            for prev_bitrate in dp[i]:
                prev_quality, prev_config = dp[i][prev_bitrate]
                
                for qp in qp_options:
                    # 计算当前选择的码率和质量（使用基础预测）
                    bitrate_inc = self._get_single_attribute_bitrate(attr, qp)
                    quality_factor = self.quality_functions[attr](qp)
                    
                    new_bitrate = prev_bitrate + bitrate_inc
                    new_quality = prev_quality * quality_factor
                    
                    # 离散化码率
                    discrete_bitrate = int(np.ceil(new_bitrate / bitrate_step)) * bitrate_step
                    
                    if discrete_bitrate <= max_bitrate:
                        if (discrete_bitrate not in dp[current_stage] or 
                            new_quality > dp[current_stage][discrete_bitrate][0]):
                            new_config = prev_config.copy()
                            new_config[attr] = qp
                            dp[current_stage][discrete_bitrate] = (new_quality, new_config)
        
        for bitrate in dp[len(attributes)]:
            if bitrate >= min_bitrate:
                config = dp[len(attributes)][bitrate][1]
                quality = self.predict_quality_basic(config)
                dp[len(attributes)][bitrate] = (quality, config)
        
        return self._filter_pareto_front(
            [(config, self.predict_quality_basic(config), float(self.calculate_bitrate(config)))
             for _, config in dp[len(attributes)].values()
             if min_bitrate <= self.calculate_bitrate(config) <= max_bitrate]
        )
        
        # 第二阶段：使用enhanced预测重新评估所有候选解
    #     print("使用enhanced预测重新评估候选解...")
    #     enhanced_solutions = []
        
    #     for bitrate in dp[len(attributes)]:
    #         if bitrate >= min_bitrate:
    #             config = dp[len(attributes)][bitrate][1]
                
    #             # 使用enhanced预测重新计算质量
    #             enhanced_quality = self.predict_quality_enhanced(config)
    #             enhanced_solutions.append((config, enhanced_quality, bitrate))
        
    #     # 第三阶段：在enhanced预测空间中进行局部优化
    #     print("在enhanced预测空间中进行局部优化...")
    #     optimized_solutions = self._local_optimize_pareto_enhanced(enhanced_solutions)
        
    #     return self._filter_pareto_front(optimized_solutions)

    # def _local_optimize_pareto_enhanced(self, pareto_candidates, max_iterations=5):
    #     """在enhanced预测空间中对Pareto候选进行局部优化"""
    #     optimized_solutions = []
        
    #     for config, quality, bitrate in pareto_candidates:
    #         best_config = config
    #         best_quality = quality
            
    #         for iteration in range(max_iterations):
    #             improved = False
                
    #             # 生成邻域配置
    #             neighbors = self._generate_dp_neighbors(best_config)
                
    #             for neighbor in neighbors:
    #                 neighbor_bitrate = self.calculate_bitrate(neighbor)
                    
    #                 # 检查码率约束
    #                 if neighbor_bitrate <= bitrate:  # 保持相同或更低的码率
    #                     neighbor_quality = self.predict_quality_enhanced(neighbor)
                        
    #                     if neighbor_quality > best_quality:
    #                         best_config = neighbor
    #                         best_quality = neighbor_quality
    #                         improved = True
                
    #             if not improved:
    #                 break
            
    #         optimized_solutions.append((best_config, best_quality, bitrate))
        
    #     return optimized_solutions

    def _generate_dp_neighbors(self, config):
        """为DP生成的配置生成邻域"""
        neighbors = []
        attributes = list(config.keys())
        
        # 单属性微调（±10 QP）
        for attr in attributes:
            current_qp = config[attr]
            
            # 增加QP（降低质量，减少码率）
            if current_qp < 50:
                neighbor_high = config.copy()
                neighbor_high[attr] = min(50, current_qp + 10)
                neighbors.append(neighbor_high)
            
            # 减少QP（提高质量，增加码率）
            if current_qp > 0:
                neighbor_low = config.copy()
                neighbor_low[attr] = max(0, current_qp - 10)
                neighbors.append(neighbor_low)
        
        # 质量-码率权衡：提高敏感属性质量，降低不敏感属性质量
        sensitive_attrs = ['color', 'scale']  # 根据您的数据
        insensitive_attrs = ['position_low', 'rotation']
        
        for sensitive_attr in sensitive_attrs:
            for insensitive_attr in insensitive_attrs:
                if (config[sensitive_attr] > 0 and 
                    config[insensitive_attr] < 50):
                    
                    neighbor = config.copy()
                    neighbor[sensitive_attr] = max(0, config[sensitive_attr] - 10)  # 提高敏感属性质量
                    neighbor[insensitive_attr] = min(50, config[insensitive_attr] + 10)  # 降低不敏感属性质量
                    neighbors.append(neighbor)
        
        return neighbors


    def _get_single_attribute_bitrate(self, attribute, qp):
        """获取单个属性的码率"""
        if attribute == 'position_low':
            position_high = 49.8 * self.channel_multipliers['position_high']
            position_low = self.bitrate_functions['position_low'](qp) * self.channel_multipliers['position_low']
            return position_high + position_low
        else:
            data_key = self.config_mapping[attribute]
            return self.bitrate_functions[data_key](qp) * self.channel_multipliers[data_key]

    def optimize_single_target(self, target_bitrate):
        """针对单个目标码率进行优化"""
        def objective(qp_vector):
            config = {
                'position_low': qp_vector[0], 'color': qp_vector[1],
                'opacity': qp_vector[2], 'scale': qp_vector[3], 'rotation': qp_vector[4]
            }
            return -self.predict_quality(config)

        def constraint(qp_vector):
            config = {
                'position_low': qp_vector[0], 'color': qp_vector[1],
                'opacity': qp_vector[2], 'scale': qp_vector[3], 'rotation': qp_vector[4]
            }
            return target_bitrate - self.calculate_bitrate(config)

        initial_guess = [20, 20, 20, 20, 20]
        bounds = [(0, 50)] * 5
        constraints = {'type': 'ineq', 'fun': constraint}
        
        result = minimize(objective, initial_guess, bounds=bounds, 
                         constraints=constraints, method='SLSQP')
        
        if result.success:
            continuous_config = dict(zip(['position_low', 'color', 'opacity', 'scale', 'rotation'], 
                                       result.x))
            discrete_config = {k: int(round(v / 10) * 10) for k, v in continuous_config.items()}
            discrete_config = {k: min(50, max(0, v)) for k, v in discrete_config.items()}
            
            if self.calculate_bitrate(discrete_config) <= target_bitrate:
                return discrete_config
        return None

    def _filter_pareto_front(self, solutions):
        """过滤得到Pareto前沿"""
        if not solutions:
            return []
        
        # 按码率排序
        solutions.sort(key=lambda x: x[2])
        
        pareto_front = []
        best_quality = -float('inf')
        
        for config, quality, bitrate in solutions:
            if quality >= best_quality:
                pareto_front.append((config, quality, bitrate))
                best_quality = quality
            # else:
            #     pareto_front.append((config, best_quality, bitrate))
        
        return pareto_front

    def plot_pareto_frontier(self, pareto_front, save_path=None):
        """绘制Pareto前沿"""
        if not pareto_front:
            print("没有Pareto前沿数据可绘制")
            return
        
        bitrates = [point[2] for point in pareto_front]
        qualities = [point[1] for point in pareto_front]
        configs = [point[0] for point in pareto_front]
        
        plt.figure(figsize=(10, 6))
        plt.plot(bitrates, qualities, 'bo-', linewidth=1.5, markersize=2, label='Pareto Frontier')
        
        # 标记一些关键点
        # for i, (config, quality, bitrate) in enumerate(pareto_front):
        #     if i % max(1, len(pareto_front) // 5) == 0:  # 标记约5个点
        #         qp_str = '/'.join(str(config[attr]) for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation'])
        #         plt.annotate(f'({qp_str})', (bitrate, quality), 
        #                    xytext=(10, 10), textcoords='offset points', fontsize=8)
        unified_quality_func = CubicSpline(self.unified_qp_levels, self.unified_qp_quality)
        unified_qp_test = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        unified_qualities = [unified_quality_func(qp) for qp in unified_qp_test]
        unified_bitrates = [self.calculate_bitrate({attr: qp for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation']}) 
                          for qp in unified_qp_test]
        
        plt.plot(unified_bitrates, unified_qualities, 'ro--', linewidth=1, markersize=4, 
                label='Uniform QP (Reference)')
        plt.xlabel('Bitrate (KB/frame)')
        plt.ylabel('Quality (PSNR dB)')
        plt.title('Pareto Frontier: Quality vs Bitrate Trade-off')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pareto前沿图已保存到: {save_path}")
        
        plt.show()

    def export_pareto_frontier(self, pareto_front, filename='pareto_frontier.json'):
        """导出Pareto前沿到JSON文件"""
        import json
        
        export_data = []
        for config, quality, bitrate in pareto_front:
            point_data = {
                'config': config,
                'quality_psnr': quality,
                'bitrate_kb': bitrate
            }
            export_data.append(point_data)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Pareto前沿已导出到: {filename}")

def pareto_gen(optimizer):
    print("生成Pareto前沿...")
    
    # # 方法1: 智能采样（推荐，平衡速度和质量）
    # print("\n1. 使用智能采样方法...")
    # pareto_smart = optimizer.generate_pareto_frontier(method='smart_sampling', num_points=15)
    # print(f"找到 {len(pareto_smart)} 个Pareto最优解")
    
    # # 显示结果
    # print("\nPareto前沿 (按码率排序):")
    # for i, (config, quality, bitrate) in enumerate(pareto_smart):
    #     print(f"{i+1:2d}. 码率: {bitrate:6.1f} KB, 质量: {quality:6.2f} dB, "
    #           f"配置: pos_low={config['position_low']:2d}, color={config['color']:2d}, "
    #           f"opacity={config['opacity']:2d}, scale={config['scale']:2d}, rotation={config['rotation']:2d}")
    
    # # 绘制Pareto前沿
    # optimizer.plot_pareto_frontier(pareto_smart, save_path='pareto_frontier.png')
    
    # # 导出结果
    # optimizer.export_pareto_frontier(pareto_smart, 'pareto_frontier.json')
    
    # 方法2: 动态规划（更精确但较慢）
    import time
    start_time = time.time()
    print("\n2. 使用动态规划方法...")
    pareto_dp = optimizer.generate_pareto_frontier(method='dp')
    print(f"找到 {len(pareto_dp)} 个Pareto最优解 (动态规划)")
    end_time = time.time()
    print(f"动态规划耗时: {end_time - start_time:.2f} 秒")
    
    optimizer.plot_pareto_frontier(pareto_dp, save_path=REPO_ROOT / 'assets' / 'visualization' / 'pareto_frontier_dp_ori.png')
    optimizer.plot_pareto_frontier(pareto_dp, save_path=REPO_ROOT / 'assets' / 'visualization' / 'pareto_frontier_dp_ori.pdf')
    optimizer.export_pareto_frontier(pareto_dp, REPO_ROOT / 'assets' / 'visualization' / 'pareto_frontier_dp_ori.json')
    
    # 比较两种方法
    print(f"\n方法比较:")
    # print(f"智能采样: {len(pareto_smart)} 个解")
    print(f"动态规划: {len(pareto_dp)} 个解")
    
    # 分析Pareto前沿的特性
    # if pareto_smart:
    #     min_bitrate = min(point[2] for point in pareto_smart)
    #     max_bitrate = max(point[2] for point in pareto_smart)
    #     min_quality = min(point[1] for point in pareto_smart)
    #     max_quality = max(point[1] for point in pareto_smart)
        
    #     print(f"\nPareto前沿范围:")
    #     print(f"码率范围: {min_bitrate:.1f} - {max_bitrate:.1f} KB")
    #     print(f"质量范围: {min_quality:.2f} - {max_quality:.2f} dB")
    #     print(f"质量-码率权衡比: {(max_quality - min_quality) / (max_bitrate - min_bitrate):.4f} dB/KB")


# 使用示例
if __name__ == "__main__":
    optimizer = GaussianCompressionOptimizer()
    pareto_gen(optimizer)
    # optimizer.validate_prediction_accuracy()
    test_configs = [{'position_low': 0, 'color': 0, 'opacity': 0, 'scale': 0, 'rotation': 0},
                    {'position_low': 5, 'color': 5, 'opacity': 5, 'scale': 5, 'rotation': 5},
                     {'position_low': 10, 'color': 10, 'opacity': 10, 'scale': 10, 'rotation': 10},
                     {'position_low': 15, 'color': 15, 'opacity': 15, 'scale': 15, 'rotation': 15},
                     {'position_low': 20, 'color': 20, 'opacity': 20, 'scale': 20, 'rotation': 20},
                    {'position_low': 25, 'color': 25, 'opacity': 25, 'scale': 25, 'rotation': 25},
                        {'position_low': 30, 'color': 30, 'opacity': 30, 'scale': 30, 'rotation': 30},
                        {'position_low': 35, 'color': 35, 'opacity': 35, 'scale': 35, 'rotation': 35},
                        {'position_low': 40, 'color': 40, 'opacity': 40, 'scale': 40, 'rotation': 40},
                        {'position_low': 45, 'color': 45, 'opacity': 45, 'scale': 45, 'rotation': 45},
                        {'position_low': 50, 'color': 50, 'opacity': 50, 'scale': 50, 'rotation': 50}]
    
    for test_config in test_configs:
        bitrate = optimizer.calculate_bitrate(test_config)
        quality = optimizer.predict_quality(test_config)
        print(f"测试配置码率: {bitrate:.2f} KB, 质量: {quality:.2f} dB")
    
    
    
