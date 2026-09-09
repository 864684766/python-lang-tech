# -*- coding: utf-8 -*-
from langchain_core.prompts import StringPromptTemplate, PromptTemplate
from typing import List

class CustomTemplate(StringPromptTemplate):
    input_variables: List[str] = []
    suffix: str = "test"
    
    def format(self, **kwargs):
        return super().format(**kwargs)

# Test 1: Can we access template?
print("=== Test 1: Access template ===")
c = CustomTemplate(template="hello {topic}", input_variables=["topic"])
print(f"model_dump keys: {list(c.model_dump().keys())}")
print(f"Has 'template': {hasattr(c, 'template')}")

# Test 2: What does super().format() do?
print("\n=== Test 2: Call format() ===")
try:
    result = c.format(topic="Machine Learning")
    print(f"format() returned: {result}")
except Exception as e:
    print(f"Error in format(): {e}")

# Test 3: Compare with PromptTemplate
print("\n=== Test 3: PromptTemplate behavior ===")
pt = PromptTemplate(template="hello {topic}", input_variables=["topic"])
print(f"PromptTemplate has template: {hasattr(pt, 'template')}")
print(f"PromptTemplate template value: {pt.template}")
result = pt.format(topic="ML")
print(f"PromptTemplate.format() result: {result}")

# Test 4: Check if StringPromptTemplate has _prompt attribute
print("\n=== Test 4: Check private attrs ===")
for attr in dir(c):
    if 'prompt' in attr.lower() or 'temp' in attr.lower():
        print(f"  {attr}")
