from .ldp_agent import LDPAgent as agent
import jax.numpy as jnp
import jax
import flax 
import orbax.checkpoint as ckpt
from functools import partial
from typing import Any

class DAggerAgent(flax.struct.PyTreeNode):

    expert_model: Any
    student_model: Any
    config: dict = flax.struct.field(pytree_node=False)
    

    beta: float = 1.0  
    dagger_iteration: int = 0
    
    @classmethod
    def create_expert_model(cls, rng, batch, shape_meta, expert_ckpt_path, **agent_kwargs):
        """Create and load frozen expert model from checkpoint"""
        # Create expert agent
        expert_rng, rng = jax.random.split(rng)
        expert = agent.create(expert_rng, batch, shape_meta, **agent_kwargs)
        
        # Load checkpoint
        ckpter = ckpt.PyTreeCheckpointer()
        raw_restored = ckpter.restore(expert_ckpt_path)
        
        # Load planner params
        if 'planner_params' in raw_restored and expert.use_planner:
            expert_planner_state = expert.planner_state.replace(
                params=raw_restored['planner_params']
            )
            expert = expert.replace(planner_state=expert_planner_state)
        
        print(f"Loaded expert model from {expert_ckpt_path}")
        return expert
    
    @classmethod
    def create_student_model(cls, rng, batch, shape_meta, **agent_kwargs):
        """Create trainable student model with same architecture as expert"""
        student_rng, rng = jax.random.split(rng)
        student = agent.create(student_rng, batch, shape_meta, **agent_kwargs)
        print("Created student model with same architecture as expert")
        return student
    
    @classmethod
    def create(cls, rng, batch, shape_meta, expert_ckpt_path, 
               beta_schedule='linear', dagger_iterations=10, **agent_kwargs):
        """Create DAgger agent with expert and student models"""
        
        # Create expert model (frozen)
        expert_rng, student_rng = jax.random.split(rng)
        expert = cls.create_expert_model(expert_rng, batch, shape_meta, 
                                       expert_ckpt_path, **agent_kwargs)
        
        # Create student model (trainable, same architecture)
        student = cls.create_student_model(student_rng, batch, shape_meta, **agent_kwargs)
        
        # DAgger config
        config = {
            'beta_schedule': beta_schedule,
            'dagger_iterations': dagger_iterations,
            'expert_ckpt_path': expert_ckpt_path,
            'noise_fusion': True,  
        }
        
        return cls(
            expert_model=expert,
            student_model=student,
            beta=1.0,
            dagger_iteration=0,
            config=config
        )
    
    def fused_planner_loss(self, student_params, batch, rng, obs_horizon):
       
        obs_emb = self.student_model.get_obs_cond(batch['obs'])

        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.randint(t_rng, (obs_emb.shape[0],), 0, 
                              self.student_model.config['planner_n_diffusion_steps'])
        next_obs_emb = obs_emb[:, obs_horizon:]
        noise = jax.random.normal(noise_rng, shape=next_obs_emb.shape)
        noisy_next_obs_emb = self.student_model.planner_noise_scheduler.add_noise(
            self.student_model.planner_noise_state, next_obs_emb, noise, t)
        
        obs_cond = obs_emb[:, :obs_horizon, ...].reshape(obs_emb.shape[0], -1)
        
        # Student 
        student_pred_noise = self.student_model.planner_state.apply_fn(
            {"params": student_params}, noisy_next_obs_emb, t, obs_cond)
        
        # Expert 
        expert_obs_emb = self.expert_model.get_obs_cond(batch['obs']) 
        expert_obs_cond = expert_obs_emb[:, :obs_horizon, ...].reshape(expert_obs_emb.shape[0], -1)
        expert_pred_noise = self.expert_model.planner_state.apply_fn(
            {"params": self.expert_model.planner_state.params}, 
            noisy_next_obs_emb, t, expert_obs_cond)
        expert_pred_noise = jax.lax.stop_gradient(expert_pred_noise)  # 冻结梯度
        
   
        fused_pred_noise = self.beta * expert_pred_noise + (1.0 - self.beta) * student_pred_noise
        
      
        loss = jnp.mean((fused_pred_noise - noise) ** 2)
        
 
        expert_loss = jnp.mean((expert_pred_noise - noise) ** 2)
        student_loss = jnp.mean((student_pred_noise - noise) ** 2)
        noise_diff = jnp.mean((expert_pred_noise - student_pred_noise) ** 2)
        
        metrics = {
            'plan_loss': loss,
            'expert_plan_loss': expert_loss,
            'student_plan_loss': student_loss,
            'plan_noise_diff': noise_diff,
            'beta': self.beta
        }
        
        return loss, metrics
    

    
    def fused_loss(self, student_params, batch, rng, use_planner, obs_horizon):

        from utils.data_utils import postprocess_batch
        batch = postprocess_batch(batch, self.student_model.obs_normalization)
        batch['obs'] = self.student_model.vae_encode(batch['obs'])
        
        total_loss = 0
        metrics = {}
        
        if use_planner:
            rng, plan_rng = jax.random.split(rng)
            plan_loss, plan_metrics = self.fused_planner_loss(
                student_params['planner'], batch, plan_rng, obs_horizon)
            plan_loss = self.student_model.alpha_planner * plan_loss
            total_loss += plan_loss
            metrics.update(plan_metrics)
        
        metrics['loss'] = total_loss
        metrics['dagger/beta'] = self.beta
        metrics['dagger/iteration'] = self.dagger_iteration
        
        return total_loss, metrics
    
    def update(self, batch, rng, step):
 
        use_planner = bool(self.student_model.use_planner) and step % self.student_model.config['update_planner_every'] == 0
        update_planner = self.student_model.config['update_planner_until'] < 0 or step < self.student_model.config['update_planner_until']
        update_planner = update_planner and step >= self.student_model.config['update_planner_after']
        use_planner = use_planner and update_planner
        
        return self.update_step(batch, rng, use_planner, 
                               self.student_model.config['obs_horizon'])
    
    @partial(jax.jit, static_argnames=('use_planner', 'obs_horizon'))
    def update_step(self, batch, rng, use_planner, obs_horizon):
        rng, g_rng = jax.random.split(rng)
        
     
        student_params = {}
        if use_planner:
            student_params['planner'] = self.student_model.planner_state.params
   
        grads, metrics = jax.grad(self.fused_loss, has_aux=True)(
            student_params, batch, g_rng, use_planner, obs_horizon)
        
    
        new_student = self.student_model
        
        if use_planner:
            new_planner_state = self.student_model.planner_state.apply_gradients(grads=grads['planner'])
            new_student = new_student.replace(planner_state=new_planner_state)
            metrics["planner_lr"] = self.student_model.lr_schedule(self.student_model.planner_state.step)
            metrics["planner_step"] = self.student_model.planner_state.step
        
        return self.replace(student_model=new_student), metrics
    
    def update_beta(self, iteration):
       
        if self.config['beta_schedule'] == 'linear':
            new_beta = max(0.0, 1.0 - iteration / self.config['dagger_iterations'])
        elif self.config['beta_schedule'] == 'exponential':
            new_beta = 0.95 ** iteration
        else:
            new_beta = 1.0
        
        return self.replace(beta=new_beta, dagger_iteration=iteration)
    
    def sample(self, batch, rng):

        return self.student_model.sample(batch, rng)
    
    def sample_action(self, batch, rng):
     
        return self.student_model.sample_action(batch, rng)
    
    def get_expert_actions(self, batch, rng):

        expert_actions, _ = self.expert_model.sample_action(batch, rng)
        return jax.lax.stop_gradient(expert_actions)
    
    def get_student_actions(self, batch, rng):
 
        return self.student_model.sample_action(batch, rng)
    
    def get_metrics(self, batch, rng):
     
        expert_rng, student_rng = jax.random.split(rng)
        

        metrics = self.student_model.get_metrics(batch, student_rng)
        

        expert_actions = self.get_expert_actions(batch, expert_rng)
        student_actions = self.get_student_actions(batch, student_rng)
        
        metrics['dagger/action_mse'] = jnp.mean(jnp.square(expert_actions - student_actions))
        metrics['dagger/action_l1'] = jnp.mean(jnp.abs(expert_actions - student_actions))
        metrics['dagger/beta'] = self.beta
        metrics['dagger/iteration'] = self.dagger_iteration
        
        return metrics
    
    def get_params(self):
 
        return {
            'student_params': self.student_model.get_params(),
            'expert_params': self.expert_model.get_params(),  # 用于保存，但不训练
            'dagger_state': {
                'beta': self.beta,
                'iteration': self.dagger_iteration,
            }
        }





#class method dagger 
