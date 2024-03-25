import ast

class Parser:
	def parse_study_name(study_name):
		if '=' in study_name:
			name, right = study_name.split('=')
			split_result = ast.literal_eval(right)
		else:
			name, right = study_name.split('#',maxsplit=1)
			split_result = list(right.strip('#').split(','))
					
		if len(split_result)==1:
			split_result.append(None)
			
		return split_result
