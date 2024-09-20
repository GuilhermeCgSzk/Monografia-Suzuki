from abc import ABC,abstractmethod

class Name:
	@abstractmethod
	def name(self):
		pass
		
	def final_name(self):
		return self.name()
	
	def mappings(self):
		return {self.name(): self.name()}
