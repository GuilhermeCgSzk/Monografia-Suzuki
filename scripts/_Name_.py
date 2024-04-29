from abc import ABC,abstractmethod

class Name:
	@abstractmethod
	def name(self):
		pass
	
	def mappings(self):
		return {self.name(): self.name()}
