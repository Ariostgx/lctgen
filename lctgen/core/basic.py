import abc
import os

# --------------------------- Base abstract model --------------------------- #

class BasicLLM(abc.ABC):
	def __init__(self, config):
		self.config = config
		self.base_prompt = None
	
	@abc.abstractmethod
	def prepare_prompt(self, query, base_prompt):
		'''
		prepare real prompt query with base_prompt
		'''
		pass

	@abc.abstractmethod
	def llm_query(self, extended_prompt):
		'''
		perform llm query call
		'''
		pass

	@abc.abstractmethod
	def post_process(self, response):
		'''
		post process of llm query response
		'''
		pass

	def forward(self, query, base_prompt=None):
		if base_prompt is None:
			base_prompt = self.base_prompt
		
		extended_prompt = self.prepare_prompt(query, base_prompt)
		response = self.llm_query(extended_prompt)
		result = self.post_process(response)

		return result
		