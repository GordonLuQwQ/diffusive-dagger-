import h5py
import numpy as np
import random
import jax
import copy
from .robomimic_latent_data import RobomimicLatentDataset, RobomimicData  # 继承你现有的

class DAggerRobomimicLatentDataset(RobomimicLatentDataset):
    """精简版DAgger数据集：只添加重标注功能"""
    def __init__(self, suboptimal_hdf5_path=None, suboptimal_latent_path=None, **kwargs):
    # 先设置 suboptimal 相关属性，然后再调用父类初始化
        self.suboptimal_hdf5_path = suboptimal_hdf5_path
        self.suboptimal_latent_path = suboptimal_latent_path
        self._suboptimal_hdf5_file = None
        self._suboptimal_latent_file = None
        
        # 重标注数据存储
        self.relabeled_demos = []
        
        # 然后初始化原有的数据集
        super().__init__(**kwargs)
        
        # 加载suboptimal demos列表
        if suboptimal_hdf5_path:
            self.suboptimal_demos = list(self.suboptimal_hdf5_file["data"].keys())
            print(f"Loaded {len(self.suboptimal_demos)} suboptimal demos for DAgger")
    
    def relabel_with_expert(self, expert_agent, n_episodes: int, rng):
        """核心功能：用专家重标注suboptimal数据"""
        if not hasattr(self, 'suboptimal_demos') or not self.suboptimal_demos:
            print("No suboptimal data available")
            return

        # 随机选择要重标注的episodes
        episodes_to_relabel = random.sample(
            self.suboptimal_demos, 
            min(n_episodes, len(self.suboptimal_demos))
        )

        print(f"Relabeling {len(episodes_to_relabel)} episodes...")

        for demo_name in episodes_to_relabel:
            # 从suboptimal数据获取观察序列
            demo_data = self.suboptimal_hdf5_file[f'data/{demo_name}']
            episode_length = demo_data.attrs["num_samples"]
            
            # 准备重标注的episode
            relabeled_actions = []
            
            # 为每个时间步获取专家动作
            for t in range(episode_length):
                # 构建观察批次
                obs_batch = {}
                for key in self.obs_keys:
                    if key == 'optimal':
                        obs_batch[key] = np.ones((1, self.n_frame_stack, 1))
                    elif 'latent' in key and self.suboptimal_latent_file:
                        # 从latent文件获取
                        obs_data = self.suboptimal_latent_file['data'][demo_name]['latent'][key.replace('latent_', '')][:]
                        obs_seq = self._get_obs_sequence(obs_data, t)
                        obs_batch[key] = np.expand_dims(obs_seq, 0)
                    elif key in demo_data['obs']:
                        # 从HDF5文件获取
                        obs_data = demo_data['obs'][key][:]
                        obs_seq = self._get_obs_sequence(obs_data, t)
                        obs_batch[key] = np.expand_dims(obs_seq, 0)

                # 专家预测动作
                step_rng, rng = jax.random.split(rng)
                expert_action, _ = expert_agent.sample_action({'obs': obs_batch}, step_rng)
                expert_action = jax.device_get(expert_action)[0]
                
                # 如果是序列，只取第一个
                if len(expert_action.shape) > 1:
                    expert_action = expert_action[0]
                
                relabeled_actions.append(expert_action)

            # 存储重标注的demo（简化格式）
            relabeled_demo = {
                'actions': np.array(relabeled_actions),
                'demo_name': demo_name,
                'length': episode_length
            }
            self.relabeled_demos.append(relabeled_demo)

        # 重新构建数据
        self._rebuild_data()
        print(f"Relabeling completed! Added {len(episodes_to_relabel)} demos. Total: {len(self.demos)} demos")

    def _get_obs_sequence(self, obs_data, t):
        """获取考虑frame_stack的观察序列"""
        start_t = max(0, t - self.n_frame_stack + 1)
        end_t = t + 1
        obs_seq = obs_data[start_t:end_t]
        
        # 填充到frame_stack长度
        while len(obs_seq) < self.n_frame_stack:
            obs_seq = np.concatenate([obs_seq[:1], obs_seq], axis=0)
        
        return obs_seq

    def _rebuild_data(self):
        """重建数据：原有数据 + 重标注数据"""
        # 调用原有的weld_demos获取基础数据
        original_data = super().weld_demos()
        
        # 添加重标注数据
        if self.relabeled_demos:
            relabeled_actions = []
            relabeled_obs = {key: [] for key in self.obs_keys}
            
            for demo in self.relabeled_demos:
                # 添加actions
                actions = demo['actions']
                dummy_action = np.expand_dims(actions[-1], 0)  # 添加最后一个动作
                actions = np.concatenate([actions, dummy_action], axis=0)
                relabeled_actions.append(actions)
                
                # 添加obs（从suboptimal数据复制，除了actions）
                demo_name = demo['demo_name']
                demo_data = self.suboptimal_hdf5_file[f'data/{demo_name}']
                
                for key in self.obs_keys:
                    if key == 'optimal':
                        obs = self.optimal * np.ones((demo['length'] + 1, 1))
                        relabeled_obs[key].append(obs)
                    elif 'latent' in key and self.suboptimal_latent_file:
                        obs = self.suboptimal_latent_file['data'][demo_name]['latent'][key.replace('latent_', '')][:]
                        last_obs = np.expand_dims(obs[-1], 0)
                        obs = np.concatenate([obs, last_obs], axis=0)
                        relabeled_obs[key].append(obs)
                    elif key in demo_data['obs']:
                        obs = demo_data['obs'][key][:]
                        last_obs = np.expand_dims(obs[-1], 0)
                        obs = np.concatenate([obs, last_obs], axis=0)
                        relabeled_obs[key].append(obs)

            # 合并原有数据和重标注数据
            self.data = {}
            self.data["actions"] = np.concatenate([original_data["actions"]] + relabeled_actions, axis=0)
            
            for key in self.obs_keys:
                original_obs = [original_data[key]] if len(original_data[key]) > 0 else []
                new_obs = original_obs + relabeled_obs[key]
                if new_obs:
                    self.data[key] = np.concatenate(new_obs, axis=0)
                else:
                    self.data[key] = np.array([])

            # 更新demos列表和索引
            self.demos = self.demos + [f"relabeled_{i}" for i in range(len(self.relabeled_demos))]
            self.n_demos = len(self.demos)
            self.total_n_sequences = len(self.data["actions"])
            
            # 简化索引（直接映射到数据位置）
            self._index_to_demo_id = {i: f"seq_{i}" for i in range(self.total_n_sequences)}
        else:
            self.data = original_data

    @property
    def suboptimal_hdf5_file(self):
        """延迟加载suboptimal HDF5文件"""
        if self._suboptimal_hdf5_file is None and self.suboptimal_hdf5_path:
            self._suboptimal_hdf5_file = h5py.File(self.suboptimal_hdf5_path, 'r', 
                                                 swmr=self.hdf5_use_swmr, libver='latest')
        return self._suboptimal_hdf5_file

    @property  
    def suboptimal_latent_file(self):
        """延迟加载suboptimal latent文件"""
        if self._suboptimal_latent_file is None and self.suboptimal_latent_path:
            self._suboptimal_latent_file = h5py.File(self.suboptimal_latent_path, 'r',
                                                   swmr=self.hdf5_use_swmr, libver='latest')
        return self._suboptimal_latent_file

    def close_and_delete_hdf5_handle(self):
        """关闭所有文件句柄"""
        super().close_and_delete_hdf5_handle()
        if self._suboptimal_hdf5_file is not None:
            self._suboptimal_hdf5_file.close()
        if self._suboptimal_latent_file is not None:
            self._suboptimal_latent_file.close()
        self._suboptimal_hdf5_file = None
        self._suboptimal_latent_file = None


class DAggerRobomimicData(RobomimicData):
    """精简版DAgger数据类：只添加suboptimal数据路径"""
    
    def __init__(self, suboptimal_train_path=None, suboptimal_train_latent_path=None, **kwargs):
        super().__init__(**kwargs)
        self.suboptimal_train_path = suboptimal_train_path
        self.suboptimal_train_latent_path = suboptimal_train_latent_path

    @property
    def train_dataset(self):
        """重写train_dataset以支持DAgger"""
        if self._train_dataset is None:
            kwargs = self.ds_kwargs.copy()
            kwargs.update({
                'hdf5_path': self.train_path,
                'latent_path': self.train_latent_path, 
                'n_overfit': self.train_n_episode_overfit,
                'suboptimal_hdf5_path': self.suboptimal_train_path,
                'suboptimal_latent_path': self.suboptimal_train_latent_path,
            })
            self._train_dataset = DAggerRobomimicLatentDataset(**kwargs)
        return self._train_dataset
    def relabel_with_expert(self, expert_model, n_episodes, rng):
        """代理到数据集的重标注方法"""
        if hasattr(self.train_dataset, 'relabel_with_expert'):
            self.train_dataset.relabel_with_expert(expert_model, n_episodes, rng)
            # 清空缓存，强制重新创建数据加载器
            self._train_dataset = None
