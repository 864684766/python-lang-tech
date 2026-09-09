"""
LCEL（LangChain Expression Language）管道详解 Demo

本文件演示如何用 |（管道操作符）把多个 Runnable 组件串成一条完整的链，
以及三个核心工具类：RunnableParallel、RunnablePassthrough、RunnableBranch。

适合初学者：每一步都有详细的中文注释和图解。

运行前提：
  pip install langchain langchain-openai
  export ZHIPUAI_API_KEY="你的智谱 API Key"
"""
import os
from numbers import Number
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_zhipuai_dev.chat import ChatZhipuAI

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan

def demo_simple_chain():
    """
    这是理解 LCEL 的第一步：用 | 符号像搭积木一样连接组件。

    【核心概念】什么是 LCEL？
    ──────────────────────
    LCEL 的全称是「LangChain Expression Language」，它让你可以用非常简洁的
    语法将各种 AI 组件拼成一条流水线。| 就是水流的管道，数据从左流向右。

    【本例流程图解】
    
        {"joke": "..."}                    ← 你传入一个字典
              │
              ▼
    ┌────────────────────────────┐
    │   ChatPromptTemplate       │  ① 模板接收字典，{subject} 被替换
    │   「讲一个关于 {subject}    │     生成格式化消息列表
    │      的笑话」               │
    └──────────┬─────────────────┘
               │  （消息列表 流入下一个节点）
               ▼
    ┌────────────────────────────┐
    │   智谱 GLM 大模型           │  ② 收到消息，思考并生成回答
    │   .invoke()                │     输出 AIMessage 对象
    └──────────┬─────────────────┘
               │  （AIMessage 流入下一步）
               ▼
    ┌────────────────────────────┐
    │   StrOutputParser          │  ③ 剥掉 AIMessage 外壳，取出 .content
    │                            │     输出纯字符串
    └──────────┬─────────────────┘
               │
               ▼
        "哈哈哈..."              ← 最终拿到干净的字符串

    【为什么要加 StrOutputParser？】
    模型默认返回的是 AIMessage 对象，带有角色、内容、元信息等包装层。
    很多下游任务只需要文字本身，所以用一个 Parser 把它拆开。
    """
    print('\n' + '=' * 65)
    print('Demo 1: 最简单的一条 LCEL 管道 — 三步串联')
    print('=' * 65)
    chain = ChatPromptTemplate.from_messages([('system', '你是一个废话大师。'), ('human', '请讲一个关于 {subject} 的废话，让人一看就想睡觉,500字左右。')]) | get_model() | StrOutputParser()
    print('\n📦 管道结构: [提示词模板] → [GLM 模型] → [文本解析]\n')
    joke = chain.stream({'subject': '程序员'})
    full_answer = f'😄 笑话:'
    for content in joke:
        full_answer += content
        print(full_answer, end='', flush=True)
    print('💡 要点:')
    print('   · | 就像水管，前一节的出口自动接到后一节的入口')
    print('   · invoke() 只需传入模板需要的变量（这里是 subject）')
    print('   · 整个链还是一个 Runnable，可以再用 .stream() / .batch()')

def demo_runnable_parallel():
    """
    RunnableParallel 让你在一次调用中同时执行多条子链。

    【什么时候需要 Parallel？】

    场景举例：你想让模型同时做两件事——
      ① 翻译一段英文为中文
      ② 判断这段英文的情感倾向（正面/负面/中性）

    如果不用 Parallel，你需要写两次 invoke()，代码臃肿且可能顺序出错。
    用了 Parallel，两条分支会同时启动、各自运行、最后汇总为一个字典。

    【工作流程图】

          输入文本
             │
       ┌─────┴─────┐
       ▼            ▼
   [翻译链]      [情感链]        ← 两条分支完全独立
       │            │
       └─────┬─────┘
             ▼
    {"translation": "...",     ← 合并后的字典
     "sentiment": "..."}

    【重要细节】
    Parallel 里的 key 名字（"翻译"/"情感分析"）是你自己定的，会作为最终字典的键名。
    """
    print('\n' + '=' * 65)
    print('Demo 2: RunnableParallel — 一次调用，双线并行')
    print('=' * 65)
    translator = get_model(temperature=0.3)
    sentiment_analyzer = get_model(temperature=0.1)
    translate_chain = ChatPromptTemplate.from_template('请将以下英文翻译成自然流畅的简体中文，只输出译文不要解释：\n{text}') | translator | StrOutputParser()
    sentiment_chain = ChatPromptTemplate.from_template('判断下面这段话的情绪倾向，只用一个词回答：正面/负面/中性\n{text}') | sentiment_analyzer | StrOutputParser()
    parallel_chain = RunnableParallel({'翻译': translate_chain, '情绪判定': sentiment_chain})
    text_to_process = 'I absolutely love this product! It has made my life so much easier.'
    print(f'\n📨 待处理文本: "{text_to_process}"')
    print('\n🔄 正在同时执行 翻译 和 情绪判定 ...\n')
    results = parallel_chain.invoke({'text': text_to_process})
    print(f"📝 翻译结果: {results['翻译']}")
    print(f"😊 情绪判定: {results['情绪判定']}")
    print('\n💡 要点:')
    print('   · RunnableParallel 内部的子链并行执行，节省总时间')
    print('   · 输出的 key 名称由你自己定义（字典键）')
    print('   · 每条子链可以是任意复杂的 Runnable（单步或多步管道）')

def demo_runnable_passthrough():
    """
    RunnablePassthrough 是一个「透明通道」或「增强站」。

    它有两个超实用功能：

    ▸ 功能 A：原封不动地让数据通过（不改变输入）
    ▸ 功能 B：在原有数据基础上追加新字段

    【类比】
    想象快递包裹经过一个分拣站：
    - Passthrough 不做任何修改就放过去（功能A）
    - 或者贴上一张新的标签再继续发货（功能B）

    【为什么需要透传？】
    有些情况下，模型需要的输入不是你最初拿到的原始数据格式。
    你需要把原始数据和加工好的数据拼在一起交给下一环节。
    Passthrough 就是这个拼图胶水。

    【典型用法场景】
    RAG（检索增强生成）是最经典的使用场景：
    用户提问 → 检索知识库 → 把【问题+知识片段】一起送给模型

    【本例流程图】

    Step 1                    Step 2                  Step 3
    ┌────────────┐  ┌─────────────────────┐  ┌─────────────────┐
    │  搜索摘要   │→│ Passthrough(enhance) │→│    聊天模型       │
    │  得到 dict  │  │  加入 original_query│  │  综合回答         │
    └────────────┘  └─────────────────────┘  └─────────────────┘
                           │                        │
                           ▼                        ▼
                      enhanced_dict          final_answer_string
    """
    print('\n' + '=' * 65)
    print('Demo 3: RunnablePassthrough — 透传 & 给数据加点料')
    print('=' * 65)
    model = get_model()
    passthrough_with_lambda = RunnablePassthrough.assign(analysis=lambda x: f"[附加信息] 你问的是:{x['query']}, 我会结合已知知识来回答。")
    rag_style_chain = passthrough_with_lambda | ChatPromptTemplate.from_template('【用户问题】\n{query}\n\n【已有上下文】\n{context}\n\n【附加分析】\n{analysis}\n\n请基于以上全部信息回答：') | model | StrOutputParser()
    input_data = {'query': 'Photosynthesis 是什么过程？', 'context': '光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖和氧气的生物化学过程。'}
    print('📨 用户输入的问题:')
    print(f"   查询: {input_data['query']}")
    print(f"   上下文: {input_data['context'][:30]}...")
    print("\n🚧 Passthrough 会自动注入 'analysis' 字段 ...\n")
    answer = rag_style_chain.invoke(input_data)
    print(f'🤖 模型回答:\n{answer}\n')
    print('--- 用法 B: 纯净透传 ---\n')
    pure_passthrough = RunnablePassthrough()
    test_input = {'name': '小明', 'age': 18}
    output = pure_passthrough.invoke(test_input)
    print(f'输入: {test_input}')
    print(f'输出: {output}')
    print('(可以看到输入和输出一模一样，这就是透传)\n')
    print('💡 要点:')
    print('   · .assign(new_field=lambda x: ...) 可以在原数据上追加字段')
    print('   · 追加的新字段会自动参与后续 Prompt 模板的渲染')
    print('   · 这是构建 RAG 链路的核心技巧之一')

def demo_runnable_branch():
    """
    RunnableBranch 是一条智能交通指挥中心 🚦

    它的功能是：根据输入数据的特征，决定应该走哪条路径。
    本质上就是一个结构化的 if/elif/else 语句，但好处是可以无缝嵌入管道。

    【现实场景举例】
    用户的输入可能是：
    - 编程问题 → 走「技术解答」分支
    - 情感倾诉 → 走「共情安慰」分支
    - 其他杂项 → 走「通用助手」分支

    Branch 会自动判断该进哪条路，不需要你在外面写一堆 if。

    【Branch 的定义方式】

    RunnableBranch(
        [条件1, 满足时运行的链],
        [条件2, 满足时运行的链],
        ...
        [True, 兜底链]        ← 最后一个条件永远是 True，相当于 else
    )

    【⚠️ 重要提醒】
    条件本身也是 Runnable！它可以是一个简单的 Lambda 函数，
    也可以是一条包含 LLM 调用的复杂链（例如让模型判断意图）。
    """
    print('\n' + '=' * 65)
    print('Demo 4: RunnableBranch — 根据输入类型自动选路')
    print('=' * 65)
    model = get_model()
    poetry_chain = ChatPromptTemplate.from_messages([('system', '你是一位才华横溢的现代诗人。请用最优美的语言写一首短诗。'), ('human', '以「{topic}」为主题写一首小诗：')]) | model | StrOutputParser()
    code_help_chain = ChatPromptTemplate.from_messages([('system', '你是一位资深软件工程师。请用简洁准确的方式解释技术问题。'), ('human', '技术问题: {topic}\n请简要回答：')]) | model | StrOutputParser()
    casual_chain = ChatPromptTemplate.from_messages([('system', '你是一位友善的聊天伙伴，轻松自然地对话即可。'), ('human', '{topic}')]) | model | StrOutputParser()

    def is_poetry_request(x):
        """检查是否含有诗歌/文学相关关键词"""
        keywords = ['诗', '诗歌', '抒情', '现代诗', '古诗']
        topic = str(x.get('topic', ''))
        return any((kw in topic for kw in keywords))

    def is_code_question(x):
        """检查是否是编程技术问题"""
        keywords = ['代码', 'python', 'bug', '函数', '类', '报错', '算法']
        topic = str(x.get('topic', '')).lower()
        return any((kw in topic for kw in keywords))

    def route(inputs: dict):
        """
        智能路由
        :param inputs: 输入内容
        :return: 返回具体的模型执行
        """
        if is_poetry_request(inputs):
            return poetry_chain
        elif is_code_question(inputs):
            return code_help_chain
        else:
            return casual_chain
    branch_router = RunnableLambda(route)
    test_cases = [{'type_label': '🎨 诗歌请求', 'topic': '请以月亮为主题写一首现代诗'}, {'type_label': '💻 代码问题', 'topic': 'Python中装饰器@的含义是什么？'}, {'type_label': '☕ 普通闲聊', 'topic': '今天天气真不错'}]
    print('\n🚦 智能路由器就绪，开始分发三类请求:\n')
    for case in test_cases:
        label = case.pop('type_label')
        original_topic = case['topic']
        print(f'{label}: 「{original_topic}」')
        result = branch_router.invoke(case)
        if '诗' in original_topic or '月' in original_topic:
            route_name = '→ 诗人链 🪶'
        elif '装饰器' in original_topic or 'python' in original_topic.lower():
            route_name = '→ 工程师链 🔧'
        else:
            route_name = '→ 闲聊链 💬'
        print(f'   {route_name}')
        print(f"   回答:\n   {' '.join(result.strip().split())}\n")
    print('💡 要点:')
    print('   · RunnableBranch 按顺序检查条件，命中第一个就走对应分支')
    print('   · 最后一项条件必须是 True，充当 else 兜底')
    print('   · 进阶玩法: 让 LLM 自己做条件判断（意图识别），实现全自动路由')

def demo_full_pipeline():
    """
    终极 Demo：把 Parallel、Passthrough、Branch、Pipeline 全部串联，
    搭建一个「多功能 AI 写作助手」。

    【完整架构】

    ┌─────────────────────────────────────────────────────────────┐
    │                       用户输入                                │
    │                     {"prompt": "...",                        │
    │                       "mode": "文章|翻译|问答"}                │
    └──────────────────┬──────────────────────────────────────────┘
                       │
               ┌───────▼────────┐
               │ RunnableBranch │ ← 根据 mode 选择处理策略
               │   （智能路由）   │
               └──┬───┬───┬─────┘
                  │   │   │
           ┌──────┘   │   └──────┐
           ▼          ▼          ▼
      [文章链]    [翻译链]    [问答链]
           │          │          │
           └──────┬───┴───┬──────┘
                  │       │
            ┌─────▼───────▼─────┐
            │ RunnableParallel  │ ← 同时对内容做质量评分
            │   {               │
            │     "正文": 上面的链│
            │     "评分": 评审链   │
            │   }               │
            └────────┬──────────┘
                     │
               ┌─────▼───────┐
               │Passthrough.assign│← 拼接元数据（耗时/字数统计）
               └───────────────┘

    ⚠️ 由于这个管线涉及多次 API 调用，Token 消耗较多，本 Demo 做了精简版。
    """
    print('\n' + '=' * 65)
    print('Demo 5: 全流程综合管道 — 智能写作助手')
    print('=' * 65)
    model = get_model()
    reviewer_model = get_model(temperature=0.1)
    writing_chain = ChatPromptTemplate.from_template('请写一篇关于「{topic}」的短文，风格：{style}，字数控制在 200 字以内。') | model | StrOutputParser()
    translation_chain = ChatPromptTemplate.from_template('请将以下内容翻译成中文，保持原文的意思和语气不变：\n{text}') | model | StrOutputParser()
    qa_chain = ChatPromptTemplate.from_template('请用通俗易懂的语言回答以下问题（不超过 150 字）：\n{question}') | model | StrOutputParser()

    def intelligence_route(x: dict):
        if x['task_type'] == '写作':
            return writing_chain
        elif x['task_type'] == '翻译':
            return translation_chain
        else:
            return qa_chain
    router = RunnableLambda(intelligence_route)
    evaluator_chain = ChatPromptTemplate.from_template('以下是 AI 生成的文本。请给它的质量打分（1-10 分）并用一句话点评。\n文本：{result}') | reviewer_model | StrOutputParser()
    full_pipeline = router | {'result': RunnablePassthrough(), 'quality_score': evaluator_chain} | (lambda x: {'任务类型': x.get('task_type', '?') if isinstance(x, dict) else '未知', '生成内容': x['result'] if isinstance(x, dict) else str(x)[:500], '质量评分': x.get('quality_score', 'N/A') if isinstance(x, dict) else 'N/A'})
    tasks = [{'task_type': '写作', 'topic': '春天的樱花', 'style': '散文诗意风'}, {'task_type': '翻译', 'text': 'The best time to plant a tree was twenty years ago. The second best time is now.'}, {'task_type': '问答', 'question': '为什么天空是蓝色的？'}]
    for i, task in enumerate(tasks, 1):
        task_type_cn = {'写作': '✍️ 创意写作', '翻译': '🌐 文本翻译', '问答': '🧠 知识问答'}[task['task_type']]
        desc = task.get('topic', task.get('text', task.get('question', '')))
        print(f"\n{'━' * 60}")
        print(f'任务 [{i}] : {task_type_cn} — {desc[:40]}...')
        print(f"{'━' * 60}")
        try:
            output = full_pipeline.invoke(task)
            print(f"\n📄 生成内容:\n{output['生成内容']}\n")
            print(f"⭐ 质量评分: {output['质量评分']}")
        except Exception as e:
            print(f'❌ 出错: {e}')
    print('\n\n💡 要点:')
    print('   · 这条管线综合了: Router(Passthrough+Branch) → Chain → Evaluator')
    print('   · 你可以自由增删中间的任何一环')
    print('   · 生产级项目通常还会加上缓存、限流、监控告警')

def show_cheat_sheet():
    """打印一张速查卡片"""
    card = '\n\n╔═══════════════════════ LCEL 速查卡片 ═══════════════════════════╗\n\n║                                                                 ║\n║  📌 管道操作符 |                                                 ║\n║     a | b | c     把 a 的输出连到 b，再连到 c                  ║\n║                                                                 ║\n║  📌 RunnableParallel({k1: chain_a, k2: chain_b})               ║\n║     同时执行多条链，结果为 {k1: result_a, k2: result_b}          ║\n║     适用: 翻译+摘要、情感+分类、正文+评分                       ║\n║                                                                 ║\n║  📌 RunnablePassthrough()                                       ║\n║     原样透传数据，配合 .assign() 可追加字段                      ║\n║     适用: RAG 注入上下文、添加元数据                              ║\n║                                                                 ║\n║  📌 RunnableLambda(intelligence_route)                      ║\n║     条件路由: 通过判断条件选择不同的方法执行                             ║\n║     适用: 意图分发、多模态选择、任务分类                           ║\n║                                                                 ║\n║  📌 PromptTemplate.from_template("...")                         ║\n║     单变量: "{var}"                                             ║\n║     多变��: ChatPromptTemplate.from_messages([...])              ║\n║                                                                 ║\n║  📌 StrOutputParser()                                           ║\n║     放在管道末尾，把 AIMessage 变成 str                           ║\n║                                                                 ║\n╚═════════════════════════════════════════════════════════════════╝\n'
    print(card)
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║                                                          ║\n║          LangChain LCEL 管道完整教程                       ║\n║          (智谱 GLM-4-Flash 驱动)                         ║\n║                                                          ║\n║  涵盖: | 串联 / Parallel / Passthrough / Branch          ║\n║                                                          ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('综合管道', '用一条 | 链串 Prompt→Model→Parser；同文件还有 Parallel/Branch 可单独调用'),
    ])
    check_api_key()
    try:
        demo_full_pipeline()
    except Exception as e:
        print(f'\n❌ 运行出错: {e}')
        print('提示：检查网络连接及 ZHIPUAI_API_KEY 是否正确。')
        raise
    show_cheat_sheet()
    print('\n🎉 LCEL 教程完成！你已经掌握了 LangChain 最核心的组合能力。\n')