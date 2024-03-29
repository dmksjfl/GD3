import copy
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[400, 300]):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_sizes=[400, 300]):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)


	def forward(self, state, action):
		if len(state.shape) == 3:
			sa = torch.cat([state, action], 2)
		else:
			sa = torch.cat([state, action], 1)

		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		q = self.l3(q)

		return q


class GD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		device,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		actor_lr=1e-3,
		critic_lr=1e-3,
		hidden_sizes=[400, 300],
		beta=0.001,
		root=3,
		alpha=0.05,
		bias=0,
		num_noise_samples=50,
		with_importance_sampling=0,
        operator = 'softmax',
	):
		self.device = device

		self.actor1 = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
		self.actor1_target = copy.deepcopy(self.actor1)
		self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=actor_lr)

		self.actor2 = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
		self.actor2_target = copy.deepcopy(self.actor2)
		self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=actor_lr)

		self.critic1 = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)

		self.critic2 = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
		self.critic2_target = copy.deepcopy(self.critic2)
		self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip

		self.beta = beta
		self.root = root
		self.alpha = alpha
		self.bias = bias
		self.operator = operator
		self.num_noise_samples = num_noise_samples
		self.with_importance_sampling = with_importance_sampling
		self.actor_loss = 0
		self.critic_loss = 0
		self.Q_his = 0
		self.distance = 0.
		self.bias_distance = 0.

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

		action1 = self.actor1(state)
		action2 = self.actor2(state)

		q1 = self.critic1(state, action1)
		q2 = self.critic2(state, action2)

		action = action1 if q1 >= q2 else action2
		self.Q_his = q1 if q1 >= q2 else q2

		return action.cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		self.train_one_q_and_pi(replay_buffer, update_q1=True, batch_size=batch_size)
		d1 = self.operator_dis
		b1 = self.bias_dis
		self.train_one_q_and_pi(replay_buffer, update_q1=False, batch_size=batch_size)
		d2 = self.operator_dis
		b2 = self.bias_dis
		self.critic_loss = (self.c1_loss+self.c2_loss)/2
		self.actor_loss = (self.a1_loss+self.a2_loss)/2
		self.distance = (torch.mean(d1)+torch.mean(d2))/2
		self.bias_distance = (torch.mean(b1)+torch.mean(b2))/2

	def generalized_activation_opertator(self, beta, root, alpha, bias, importance_sampling = True, operator = 'exponential'):
		if operator == 'exponential':
			def exponential_operator(q_vals, noise_pdf=None):
				max_q_vals = torch.max(q_vals, 1, keepdim=True).values
				norm_q_vals = beta * (q_vals - max_q_vals) + bias
				e_beta_normQ = torch.pow(root, norm_q_vals)
				#print(e_beta_normQ)
				Q_mult_e = q_vals * e_beta_normQ

				numerators = Q_mult_e
				denominators = e_beta_normQ

				if importance_sampling:
					numerators /= noise_pdf
					denominators /= noise_pdf

				sum_numerators = torch.sum(numerators, 1)
				sum_denominators = torch.sum(denominators, 1)

				exponential_q_vals = sum_numerators / sum_denominators

				exponential_q_vals = torch.unsqueeze(exponential_q_vals, 1)
				return exponential_q_vals
			return exponential_operator

		elif operator == 'ReLu':
			def relu_operator(q_vals, noise_pdf=None):
				max_q_vals = torch.max(q_vals, 1, keepdim=True).values
				norm_q_vals = q_vals - max_q_vals
				e_beta_normQ = beta * norm_q_vals + bias
				Q_mult_e = q_vals * e_beta_normQ

				numerators = Q_mult_e
				denominators = e_beta_normQ

				if importance_sampling:
					numerators /= noise_pdf
					denominators /= noise_pdf

				sum_numerators = torch.sum(numerators, 1)
				sum_denominators = torch.sum(denominators, 1)

				relu_q_vals = sum_numerators / sum_denominators

				relu_q_vals = torch.unsqueeze(relu_q_vals, 1)
				return relu_q_vals
			return relu_operator
        
		elif operator == 'poly':
			def poly_operator(q_vals, noise_pdf=None):
				alpha_index = alpha
				max_q_vals = torch.max(q_vals, 1, keepdim=True).values
				norm_q_vals = q_vals - max_q_vals
				#print(norm_q_vals)
				e_beta_normQ = abs(alpha_index * norm_q_vals) ** beta + bias
				#print(e_beta_normQ)
				Q_mult_e = q_vals * e_beta_normQ

				numerators = Q_mult_e
				denominators = e_beta_normQ

				if importance_sampling:
					numerators /= noise_pdf
					denominators /= noise_pdf

				sum_numerators = torch.sum(numerators, 1)
				sum_denominators = torch.sum(denominators, 1)

				poly_q_vals = sum_numerators / sum_denominators

				poly_q_vals = torch.unsqueeze(poly_q_vals, 1)
				return poly_q_vals
			return poly_operator
        
		elif operator == 'tanh':
            
			def tanh_operator(q_vals, noise_pdf=None):
				max_q_vals = torch.max(q_vals, 1, keepdim=True).values
				norm_q_vals = q_vals - max_q_vals
				def tanh_fun(x):
					return (2*torch.exp(beta*x))/(torch.exp(beta*x) + torch.exp(-beta*x))
				e_beta_normQ = tanh_fun(norm_q_vals) + bias
				Q_mult_e = q_vals * e_beta_normQ

				numerators = Q_mult_e
				denominators = e_beta_normQ

				if importance_sampling:
					numerators /= noise_pdf
					denominators /= noise_pdf

				sum_numerators = torch.sum(numerators, 1)
				sum_denominators = torch.sum(denominators, 1)

				tanh_q_vals = sum_numerators / sum_denominators
				tanh_q_vals = torch.unsqueeze(tanh_q_vals, 1)
				return tanh_q_vals
			return tanh_operator


	def calc_pdf(self, samples, mu=0):
		pdfs = 1/(self.policy_noise * np.sqrt(2 * np.pi)) * torch.exp( - (samples - mu)**2 / (2 * self.policy_noise**2) )
		pdf = torch.prod(pdfs, dim=2)
		return pdf


	def train_one_q_and_pi(self, replay_buffer, update_q1, batch_size=100):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			if update_q1:
				next_action = self.actor1_target(next_state)
			else:
				next_action = self.actor2_target(next_state)

			noise = torch.randn(
				(action.shape[0], self.num_noise_samples, action.shape[1]), 
				dtype=action.dtype, layout=action.layout, device=action.device
			)
			noise = noise * self.policy_noise
			
			noise_pdf = self.calc_pdf(noise) if self.with_importance_sampling else None
			
			noise = noise.clamp(-self.noise_clip, self.noise_clip)

			next_action = torch.unsqueeze(next_action, 1)

			next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

			next_state = torch.unsqueeze(next_state, 1)
			next_state = next_state.repeat((1, self.num_noise_samples, 1))

			next_Q1 = self.critic1_target(next_state, next_action)
			next_Q2 = self.critic2_target(next_state, next_action)

			next_Q = torch.min(next_Q1, next_Q2)
			next_Q = torch.squeeze(next_Q, 2)
            
			gd3_next_Q = self.generalized_activation_opertator(beta = self.beta, root = self.root, alpha = self.alpha, bias = self.bias,
                                                                importance_sampling = self.with_importance_sampling,
                                                                operator = self.operator)
			activate_next_Q = gd3_next_Q(next_Q, noise_pdf)
			self.operator_dis = torch.max(next_Q, 1, keepdim=True).values - activate_next_Q
			next_Q = activate_next_Q

			target_Q = reward + not_done * self.discount * next_Q
			self.bias_dis = activate_next_Q - target_Q  ## bias = estimation by operator - true value

		if update_q1:
			current_Q = self.critic1(state, action)

			critic1_loss = F.mse_loss(current_Q, target_Q)
			#print(critic1_loss)

			self.critic1_optimizer.zero_grad()
			critic1_loss.backward()
			self.critic1_optimizer.step()

			actor1_loss = -self.critic1(state, self.actor1(state)).mean()
			
			self.actor1_optimizer.zero_grad()
			actor1_loss.backward()
			self.actor1_optimizer.step()
			self.c1_loss = critic1_loss
			self.a1_loss = actor1_loss

			for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		else:
			current_Q = self.critic2(state, action)

			critic2_loss = F.mse_loss(current_Q, target_Q)

			self.critic2_optimizer.zero_grad()
			critic2_loss.backward()
			self.critic2_optimizer.step()

			actor2_loss = -self.critic2(state, self.actor2(state)).mean()
			
			self.actor2_optimizer.zero_grad()
			actor2_loss.backward()
			self.actor2_optimizer.step()
			self.c2_loss = critic2_loss
			self.a2_loss = actor2_loss

			for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic1.state_dict(), filename + "_critic1")
		torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
		torch.save(self.actor1.state_dict(), filename + "_actor1")
		torch.save(self.actor1_optimizer.state_dict(), filename + "_actor1_optimizer")

		torch.save(self.critic2.state_dict(), filename + "_critic2")
		torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")
		torch.save(self.actor2.state_dict(), filename + "_actor2")
		torch.save(self.actor2_optimizer.state_dict(), filename + "_actor2_optimizer")

	def get_loss(self):
		return self.actor_loss, self.critic_loss

	def get_Q(self):
		return self.Q_his
	
	def get_dis(self):
		return self.distance
	
	def get_bias(self):
		return self.bias_distance

	def load(self, filename):
		self.critic1.load_state_dict(torch.load(filename + "_critic1"))
		self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
		self.actor1.load_state_dict(torch.load(filename + "_actor1"))
		self.actor1_optimizer.load_state_dict(torch.load(filename + "_actor1_optimizer"))

		self.critic2.load_state_dict(torch.load(filename + "_critic2"))
		self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
		self.actor2.load_state_dict(torch.load(filename + "_actor2"))
		self.actor2_optimizer.load_state_dict(torch.load(filename + "_actor2_optimizer"))