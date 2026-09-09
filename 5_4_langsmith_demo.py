"""
========================================================
LangChain Phase 5-4
LangSmith Tracing & Evaluation 教材级 Demo
========================================================

📚 学习目标：
1. 开启 LangSmith Tracing（环境变量配置）
2. 理解 trace 如何覆盖 LCEL 和 LangGraph
3. 了解 Dataset & Evaluation 回归测试流程
4. 使用 @traceable 装饰自定义函数

🔗 LangSmith 在 LangChain 生态中的位置：
  LCEL / LangGraph / Agent 执行
        │ 自动上报 trace（需配置环境变量）
        ▼
  LangSmith Dashboard
        · Token 用量 / 延迟 / 错误率
        · 在线 Playground 调试 Prompt
        · Dataset 回归评测

运行前提（Tracing 可选）：
  pip install langsmith langchain langchain-core zhipuai python-dotenv
  可选: LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY=lsv2_...
"""
import os
from numbers import Number
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

def is_langsmith_enabled() -> bool:
    return os.getenv('LANGCHAIN_TRACING_V2', '').lower() == 'true' and bool(os.getenv('LANGCHAIN_API_KEY'))

def demo_1_env_setup():
    print('\n' + '=' * 65)
    print('Demo 1: LangSmith 环境变量配置')
    print('=' * 65)
    env_vars = {'LANGCHAIN_TRACING_V2': os.getenv('LANGCHAIN_TRACING_V2', '(未设置)'), 'LANGCHAIN_API_KEY': '已设置' if os.getenv('LANGCHAIN_API_KEY') else '(未设置)', 'LANGCHAIN_PROJECT': os.getenv('LANGCHAIN_PROJECT', 'default（默认）'), 'LANGCHAIN_ENDPOINT': os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')}
    print('当前环境变量状态:')
    for k, v in env_vars.items():
        print(f'  · {k} = {v}')
    print('\n    启用 Tracing 的步骤（.env 文件）:\n\n      LANGCHAIN_TRACING_V2=true\n      LANGCHAIN_API_KEY=lsv2_pt_你的key\n      LANGCHAIN_PROJECT=langtest-demo\n\n    配置后，所有 LangChain/LangGraph invoke 会自动上报到 LangSmith。\n    无需修改业务代码。\n    ')

def demo_2_traced_chain():
    print('\n' + '=' * 65)
    print('Demo 2: 带 Tracing 的 LCEL 链')
    print('=' * 65)
    if not is_langsmith_enabled():
        print('⚠️  LangSmith 未配置，仍运行链但不会产生 trace')
        print('   配置 LANGCHAIN_TRACING_V2 和 LANGCHAIN_API_KEY 后重试')
    prompt = ChatPromptTemplate.from_messages([('human', '用一句话解释 {topic}')])
    chain = prompt | get_model() | StrOutputParser()
    result = chain.invoke({'topic': 'LangSmith'}, config={'tags': ['phase-5-4', 'demo'], 'metadata': {'demo_id': '5_4_traced_chain'}})
    print(f'回答: {result}')
    if is_langsmith_enabled():
        project = os.getenv('LANGCHAIN_PROJECT', 'default')
        print(f'\n✅ Trace 已上报到项目: {project}')
        print('   打开 https://smith.langchain.com 查看调用链、Token、延迟')
    print('💡 config.tags / metadata 便于在 Dashboard 中筛选')

def demo_3_traceable():
    print('\n' + '=' * 65)
    print('Demo 3: @traceable 装饰自定义逻辑')
    print('=' * 65)
    try:
        from langsmith import traceable
    except ImportError:
        print('langsmith 未安装，跳过')
        return

    @traceable(name='preprocess_question')
    def preprocess(question: str) -> str:
        """自定义预处理步骤也会出现在 trace 树中"""
        return question.strip().lower()

    @traceable(name='qa_pipeline')
    def qa(question: str) -> str:
        cleaned = preprocess(question)
        chain = ChatPromptTemplate.from_messages([('human', '{q}')]) | get_model() | StrOutputParser()
        return chain.invoke({'q': cleaned})
    answer = qa('  什么是 Tracing？  ')
    print(f'回答: {answer[:100]}...')
    print('💡 @traceable 让非 LangChain 代码也纳入同一棵 trace 树')

def demo_4_evaluation_concept():
    print('\n' + '=' * 65)
    print('Demo 4: Dataset & Evaluation 回归测试（概念）')
    print('=' * 65)
    print('\n    LangSmith Evaluation 典型工作流：\n\n    1. 创建 Dataset\n       · 在 LangSmith UI 或使用 Client 上传 (input, expected_output) 对\n\n    2. 定义被测链/Agent\n       target = prompt | model | parser\n\n    3. 运行 Evaluation\n       · 对 Dataset 中每条 input 调用 target\n       · 用 Evaluator 打分（精确匹配 / LLM-as-judge / 自定义）\n\n    4. 对比实验\n       · 修改 prompt 或换模型后重新跑\n       · Dashboard 对比两次 run 的分数、延迟、成本\n\n    Python SDK 示例（需 LANGCHAIN_API_KEY）:\n\n      from langsmith import Client\n      from langsmith.evaluation import evaluate\n\n      client = Client()\n      results = evaluate(\n          lambda inputs: chain.invoke(inputs),\n          data="my-dataset-name",\n          evaluators=[...],\n          experiment_prefix="prompt-v2",\n      )\n\n    💡 Evaluation 是生产级 Prompt 迭代的标配工具\n    ')

def demo_5_best_practices():
    print('\n' + '=' * 65)
    print('Demo 5: Observability 最佳实践')
    print('=' * 65)
    print('\n    · 开发环境: LANGCHAIN_TRACING_V2=true，全量 trace\n    · 生产环境: 采样 trace（或仅 error 路径），避免成本过高\n    · 统一 LANGCHAIN_PROJECT 命名: {app}-{env}\n    · LCEL 和 LangGraph 共用同一套 tracing，无需两套监控\n    · 用 tags 标记: ["rag", "agent", "user-123"]\n    · Prompt 调试优先用 LangSmith Playground，比本地 print 高效\n    ')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 5-4: LangSmith Tracing             ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('环境变量', 'TRACING_V2 + API_KEY + PROJECT'),
        ('带 trace 的链', 'invoke 后可在网站回放'),
        ('@traceable', '给普通函数也打点'),
        ('评测概念', 'dataset + evaluator'),
        ('最佳实践', 'tags / metadata 怎么打'),
    ])
    demo_1_env_setup()
    check_api_key()
    try:
        demo_2_traced_chain()
        demo_3_traceable()
        demo_4_evaluation_concept()
        demo_5_best_practices()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 5-4 完成！下一步: 5_5_package_management_demo.py\n')