import streamlit as st
import requests
import json
import time
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置
API_BASE_URL = os.getenv("DEFAULT_API_BASE_URL", "https://api.siliconflow.cn")
# 移除对Express后端的依赖，不再需要这些配置
# DEFAULT_PORT = os.getenv("PORT", "3000")
# BACKEND_URL = f"http://localhost:{DEFAULT_PORT}/api"

# 页面配置
st.set_page_config(
    page_title="增强型自监督提示优化系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    /* 现代化设计 */
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 500;
        color: #2563EB;
    }
    
    /* 卡片式设计 */
    .card {
        background-color: #FFFFFF;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* 更好的输出容器 */
    .output-container {
        background-color: #F9FAFB;
        border-radius: 0.75rem;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #E5E7EB;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* 评估结果颜色 */
    .better {
        border-left: 4px solid #10B981;
        background-color: #ECFDF5;
    }
    .worse {
        border-left: 4px solid #EF4444;
        background-color: #FEF2F2;
    }
    .similar {
        border-left: 4px solid #F59E0B;
        background-color: #FFFBEB;
    }
    
    /* 历史记录项 */
    .history-item {
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        border-radius: 0.5rem;
        background-color: #F9FAFB;
        transition: all 0.2s ease;
    }
    .history-item:hover {
        background-color: #F3F4F6;
    }
    .history-item.better {
        border-left: 4px solid #10B981;
    }
    .history-item.not-better {
        border-left: 4px solid #EF4444;
    }
    
    /* 提示和引导 */
    .guide-box {
        background-color: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1.2rem;
        color: #1E40AF;
    }
    .tip-box {
        background-color: #ECFDF5;
        border: 1px solid #D1FAE5;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1.2rem;
        color: #065F46;
    }
    
    /* 更好的按钮 */
    .stButton>button {
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    
    /* 流式输出区域 */
    .stream-output {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 0.5rem;
        border-left: 3px solid #3B82F6;
        min-height: 100px;
        font-family: 'Roboto Mono', monospace;
        white-space: pre-wrap;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {
            border-color: #3B82F6;
        }
        50% {
            border-color: #60A5FA;
        }
        100% {
            border-color: #3B82F6;
        }
    }
    
    /* 步骤指示器 */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
        position: relative;
    }
    .step-container::before {
        content: "";
        position: absolute;
        top: 15px;
        left: 0;
        right: 0;
        height: 2px;
        background-color: #E5E7EB;
        z-index: 1;
    }
    .step {
        background-color: #FFFFFF;
        border: 2px solid #E5E7EB;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #6B7280;
        position: relative;
        z-index: 2;
    }
    .step.active {
        background-color: #3B82F6;
        border-color: #3B82F6;
        color: white;
    }
    .step.completed {
        background-color: #10B981;
        border-color: #10B981;
        color: white;
    }
    .step-label {
        position: absolute;
        top: 35px;
        font-size: 0.8rem;
        color: #6B7280;
        text-align: center;
        width: 80px;
        left: -25px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.api_configured = False
    st.session_state.current_view = "config"
    st.session_state.current_iteration = 0
    st.session_state.max_iterations = 10
    st.session_state.samples = []
    st.session_state.current_best_prompt = ""
    st.session_state.current_best_outputs = {}
    st.session_state.new_prompt = ""
    st.session_state.new_outputs = {}
    st.session_state.evaluations = {}
    st.session_state.analysis = ""
    st.session_state.optimization_history = []
    st.session_state.is_optimizing = False
    st.session_state.available_models = []

# 新增LLM服务类，用于替代Express后端
class LLMService:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url or os.getenv("DEFAULT_API_BASE_URL", "https://api.siliconflow.cn")
    
    def get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def call_llm_api(self, endpoint, data):
        try:
            response = requests.post(
                f"{self.base_url}/{endpoint}", 
                json=data, 
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"LLM API调用失败: {str(e)}")
    
    def call_llm_api_stream(self, endpoint, data):
        """流式调用LLM API"""
        try:
            # 确保设置stream为True
            data["stream"] = True
            
            response = requests.post(
                f"{self.base_url}/{endpoint}", 
                json=data, 
                headers=self.get_headers(),
                stream=True  # 设置requests为流式请求
            )
            response.raise_for_status()
            return response
        except Exception as e:
            raise Exception(f"流式API调用失败: {str(e)}")
    
    def generate_samples(self, task_description):
        # 实现样本生成
        prompt = f"""请为以下任务生成5个测试样例，每个样例应包含问题和期望的回答标准：
        
任务描述：{task_description}

请返回JSON格式：
[
    {{"question": "问题1", "expected": "期望标准1"}},
    ...
]
"""
        response = self.call_llm_api("v1/chat/completions", {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        })
        
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            # 提取JSON部分
            json_str = content[content.find("["):content.rfind("]")+1]
            samples = json.loads(json_str)
            # 给每个样本添加ID
            for i, sample in enumerate(samples):
                sample['id'] = i + 1
            return samples
        except:
            # 如果解析失败，返回简化版样本
            return [
                {"question": "示例问题1", "expected": "期望回答标准1"},
                {"question": "示例问题2", "expected": "期望回答标准2"}
            ]
    
    def execute_prompt(self, prompt, question):
        """执行提示词获取输出"""
        full_prompt = f"{prompt}\n\n{question}"
        response = self.call_llm_api("v1/chat/completions", {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.3
        })
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def execute_prompt_stream(self, prompt, question, callback):
        """带流式输出的执行提示词"""
        full_prompt = f"{prompt}\n\n{question}"
        response = self.call_llm_api_stream("v1/chat/completions", {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.3
        })
        
        # 累积完整响应
        full_response = ""
        
        # 处理流式响应
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # 移除 "data: " 前缀
                    if data == "[DONE]":
                        break
                    
                    try:
                        json_data = json.loads(data)
                        delta = json_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            full_response += delta
                            callback(delta, full_response)  # 回调函数处理每个增量
                    except json.JSONDecodeError:
                        continue
        
        return full_response
    
    def optimize_prompt(self, current_prompt, current_output, task_description, history=""):
        """优化提示词"""
        prompt = f"""请优化以下提示词，使其更有效地完成任务：

任务描述：{task_description}

当前提示词：
{current_prompt}

当前输出示例：
{current_output}

优化历史：
{history}

请直接返回优化后的完整提示词，不要有其他解释。"""
        
        response = self.call_llm_api("v1/chat/completions", {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        })
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def evaluate_outputs(self, output_a, output_b, task_description, question):
        """评估输出质量"""
        prompt = f"""请评估以下两个输出，哪一个更好地完成了任务：

任务描述：{task_description}

问题：{question}

输出A：
{output_a}

输出B：
{output_b}

请给出详细评估，并明确指出哪个更好（A、B或相似）："""
        
        response = self.call_llm_api("v1/chat/completions", {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        })
        result = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # 简单解析结果
        if "输出A更好" in result or "A更好" in result:
            winner = "A"
        elif "输出B更好" in result or "B更好" in result:
            winner = "B"
        else:
            winner = "similar"
            
        return {
            "details": result,
            "winner": winner
        }
    
    def analyze_changes(self, old_prompt, new_prompt, task_description):
        """分析提示词变化"""
        prompt = f"""请分析以下提示词的变化，并解释这些变化如何提升了提示词的效果：

任务描述：{task_description}

旧提示词：
{old_prompt}

新提示词：
{new_prompt}

请详细说明："""
        
        response = self.call_llm_api("v1/chat/completions", {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        })
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")

# 获取可用模型列表
def get_available_models():
    # 仅返回DeepSeek系列模型
    return [
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1"
    ]

# 调用API函数已不再需要，因为使用LLMService类直接调用

# 配置API
def configure_api(api_key, base_url, models):
    try:
        # 创建LLMService实例并保存在会话状态中
        st.session_state.llm_service = LLMService(api_key=api_key, base_url=base_url)
        # 保存API配置信息
        st.session_state.api_key = api_key
        st.session_state.base_url = base_url
        st.session_state.api_configured = True
        return True
    except Exception as e:
        st.error(f"API配置失败: {str(e)}")
        return False

# 生成测试样本
def generate_samples(task_description):
    try:
        # 直接使用LLMService实例
        if "llm_service" in st.session_state:
            samples = st.session_state.llm_service.generate_samples(task_description)
            # 给每个样本添加ID
            for i, sample in enumerate(samples):
                sample['id'] = i + 1
            return samples
        else:
            st.error("请先配置API")
            return []
    except Exception as e:
        st.error(f"生成样本失败: {str(e)}")
        return []

# 执行提示词
def execute_prompt(prompt, question, use_stream=False):
    try:
        # 直接使用LLMService实例
        if "llm_service" in st.session_state:
            if use_stream:
                # 创建一个空的占位符用于显示流式输出
                output_placeholder = st.empty()
                
                # 定义回调函数处理流式输出
                def stream_callback(delta, full_response):
                    output_placeholder.markdown(full_response)
                
                # 调用流式API
                return st.session_state.llm_service.execute_prompt_stream(
                    prompt, question, stream_callback
                )
            else:
                # 使用原有的非流式API
                return st.session_state.llm_service.execute_prompt(prompt, question)
        else:
            st.error("请先配置API")
            return ""
    except Exception as e:
        st.error(f"执行提示词失败: {str(e)}")
        return ""

# 优化提示词
def optimize_prompt(current_prompt, current_output, task_description, history=""):
    try:
        # 直接使用LLMService实例
        if "llm_service" in st.session_state:
            return st.session_state.llm_service.optimize_prompt(
                current_prompt, 
                current_output, 
                task_description, 
                history
            )
        else:
            st.error("请先配置API")
            return ""
    except Exception as e:
        st.error(f"优化提示词失败: {str(e)}")
        return ""

# 评估输出
def evaluate_outputs(output_a, output_b, task_description, question):
    try:
        # 直接使用LLMService实例
        if "llm_service" in st.session_state:
            result = st.session_state.llm_service.evaluate_outputs(
                output_a, 
                output_b, 
                task_description, 
                question
            )
            # 根据winner判断返回的结果
            if result['winner'] == 'A':
                return "A更好"
            elif result['winner'] == 'B':
                return "B更好"
            else:
                return "相似"
        else:
            st.error("请先配置API")
            return "相似"
    except Exception as e:
        st.error(f"评估输出失败: {str(e)}")
        return "相似"

# 分析提示变化
def analyze_changes(old_prompt, new_prompt, task_description):
    try:
        # 直接使用LLMService实例
        if "llm_service" in st.session_state:
            return st.session_state.llm_service.analyze_changes(
                old_prompt, 
                new_prompt, 
                task_description
            )
        else:
            st.error("请先配置API")
            return ""
    except Exception as e:
        st.error(f"分析提示变化失败: {str(e)}")
        return ""

# 执行当前最佳提示词
def run_current_best_prompt():
    outputs = {}
    
    with st.spinner("正在执行当前提示词..."):
        for sample in st.session_state.samples:
            output = execute_prompt(st.session_state.current_best_prompt, sample['question'])
            outputs[sample['id']] = output
    
    st.session_state.current_best_outputs = outputs
    return outputs

# 获取优化历史摘要
def get_optimization_history_summary():
    if not st.session_state.optimization_history:
        return ""
    
    # 只取最近3次迭代的历史
    recent_history = st.session_state.optimization_history[-3:]
    
    return "\n".join([
        f"迭代{item['iteration']}: {'改进成功' if item['is_better'] else '未改进'}"
        for item in recent_history
    ])

# 判断是否应该更新最佳提示
def should_update_best_prompt(evaluations):
    better_count = 0
    worse_count = 0
    
    for sample_id, result in evaluations.items():
        if result == "B更好":
            better_count += 1
        elif result == "A更好":
            worse_count += 1
    
    # 如果有更多样本认为新提示更好，则更新
    return better_count > worse_count

# 运行优化步骤
def run_optimization_step():
    if st.session_state.current_iteration >= st.session_state.max_iterations:
        st.session_state.is_optimizing = False
        st.session_state.current_view = "results"
        return
    
    st.session_state.current_iteration += 1
    
    # 1. 生成新提示候选
    with st.spinner(f"正在执行第 {st.session_state.current_iteration} 次优化..."):
        new_prompt = optimize_prompt(
            st.session_state.current_best_prompt,
            json.dumps(st.session_state.current_best_outputs),
            st.session_state.task_description,
            get_optimization_history_summary()
        )
        
        if not new_prompt:
            st.error("优化提示词失败")
            st.session_state.is_optimizing = False
            return
        
        st.session_state.new_prompt = new_prompt
        
        # 2. 执行新提示
        new_outputs = {}
        for sample in st.session_state.samples:
            output = execute_prompt(new_prompt, sample['question'])
            new_outputs[sample['id']] = output
        
        st.session_state.new_outputs = new_outputs
        
        # 3. 评估新旧输出
        evaluations = {}
        for sample in st.session_state.samples:
            sample_id = sample['id']
            output_a = st.session_state.current_best_outputs.get(sample_id, "")
            output_b = new_outputs.get(sample_id, "")
            
            evaluation = evaluate_outputs(
                output_a,
                output_b,
                st.session_state.task_description,
                sample['question']
            )
            
            evaluations[sample_id] = evaluation
        
        st.session_state.evaluations = evaluations
        
        # 4. 分析提示变化
        analysis = analyze_changes(
            st.session_state.current_best_prompt,
            new_prompt,
            st.session_state.task_description
        )
        
        st.session_state.analysis = analysis
        
        # 5. 根据评估结果更新最佳提示
        is_better = should_update_best_prompt(evaluations)
        if is_better:
            st.session_state.current_best_prompt = new_prompt
            st.session_state.current_best_outputs = new_outputs
        
        # 记录优化历史
        st.session_state.optimization_history.append({
            "iteration": st.session_state.current_iteration,
            "prompt": new_prompt,
            "is_better": is_better,
            "analysis": analysis,
            "evaluations": evaluations
        })
    
    # 如果是自动模式，继续优化
    if st.session_state.auto_mode:
        time.sleep(1)  # 短暂延迟，让UI更新
        run_optimization_step()

# 配置视图
def show_config_view():
    st.markdown("<h1 class='main-header'>SPO+ 增强型自监督提示优化系统</h1>", unsafe_allow_html=True)
    
    # 步骤指示器
    st.markdown("""
    <div class="step-container">
        <div class="step active">
            1
            <div class="step-label">API配置</div>
        </div>
        <div class="step">
            2
            <div class="step-label">任务设置</div>
        </div>
        <div class="step">
            3
            <div class="step-label">优化过程</div>
        </div>
        <div class="step">
            4
            <div class="step-label">优化结果</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 引导提示
    st.markdown("""
    <div class="guide-box">
        <h4>🚀 欢迎使用提示词优化系统</h4>
        <p>本系统可以帮助您自动优化AI提示词，提高AI输出质量。首先，请完成API配置并设置任务需求。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 获取可用模型列表
    if not st.session_state.available_models:
        st.session_state.available_models = get_available_models()
    
    with st.form("config_form", clear_on_submit=False):
        st.markdown("<h2 class='sub-header'>API设置</h2>", unsafe_allow_html=True)
        
        # API设置卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        api_key = st.text_input("API Key", type="password", 
                                help="输入您的SiliconCloud API密钥")
        base_url = st.text_input("API Base URL", value=API_BASE_URL,
                                help="API基础URL地址，默认使用SiliconCloud")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>模型设置</h2>", unsafe_allow_html=True)
        
        # 模型设置卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box">
            <p>💡 提示：根据不同任务选择适合的模型。DeepSeek-V3适合需要快速输出的任务，DeepSeek-R1适合数学推理和精准回答。</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            optimizer_model = st.selectbox(
                "优化模型 (LLM-1)",
                options=st.session_state.available_models,
                index=0 if st.session_state.available_models else None,
                help="用于生成优化后的提示词"
            )
            
            evaluator_model = st.selectbox(
                "评估模型 (LLM-3)",
                options=st.session_state.available_models,
                index=0 if len(st.session_state.available_models) > 0 else None,
                help="用于评估不同提示词的效果"
            )
        
        with col2:
            executor_model = st.selectbox(
                "执行模型 (LLM-2)",
                options=st.session_state.available_models,
                index=0 if len(st.session_state.available_models) > 0 else None,
                help="用于执行提示词生成回答"
            )
            
            analyzer_model = st.selectbox(
                "分析模型 (LLM-4)",
                options=st.session_state.available_models,
                index=0 if len(st.session_state.available_models) > 0 else None,
                help="用于分析提示词的变化"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>任务设置</h2>", unsafe_allow_html=True)
        
        # 任务设置卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        task_description = st.text_area(
            "任务需求描述", 
            height=100,
            placeholder="例如：创建一个能够解释复杂科学概念的AI助手，使其回答准确且易于理解...",
            help="详细描述您希望AI完成的任务和期望的输出风格"
        )
        
        initial_prompt = st.text_area(
            "初始提示词", 
            height=150,
            placeholder="例如：你是一位擅长解释复杂科学概念的AI助手。当用户提出科学问题时，你应该...",
            help="输入您目前使用的提示词作为优化起点"
        )
        
        st.markdown("""
        <div class="tip-box">
            <p>💡 提示：初始提示词质量会影响最终优化效果。请尽量提供详细的任务描述和初始提示词。</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_iterations = st.number_input(
                "最大迭代次数", 
                min_value=1, 
                max_value=20, 
                value=10,
                help="系统将执行的最大优化次数"
            )
        
        with col2:
            auto_mode = st.checkbox(
                "自动模式", 
                value=True,
                help="开启后系统将自动完成全部优化过程，无需人工干预"
            )
            
            use_streaming = st.checkbox(
                "流式输出", 
                value=True,
                help="开启后将实时显示AI生成过程"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("开始优化", use_container_width=True)
        with col2:
            show_example = st.form_submit_button("加载示例", use_container_width=True)
        
        if show_example:
            # 加载示例任务和提示词
            st.session_state.example_loaded = True
            st.rerun()
        
        if submitted:
            if not api_key:
                st.error("⚠️ 请输入API Key")
                return
            
            if not task_description:
                st.error("⚠️ 请输入任务需求描述")
                return
            
            if not initial_prompt:
                st.error("⚠️ 请输入初始提示词")
                return
            
            # 保存流式输出设置
            st.session_state.use_streaming = use_streaming
            
            # 配置模型
            models = {
                "optimizer": optimizer_model,
                "executor": executor_model,
                "evaluator": evaluator_model,
                "analyzer": analyzer_model
            }
            
            # 保存配置到会话状态
            st.session_state.task_description = task_description
            st.session_state.current_best_prompt = initial_prompt
            st.session_state.max_iterations = max_iterations
            st.session_state.auto_mode = auto_mode
            
            # 配置API
            with st.spinner("正在配置API..."):
                try:
                    result = configure_api(api_key, base_url, models)
                    if result:
                        st.success("✅ API配置成功")
                        
                        # 显示进度信息
                        progress_text = "正在生成测试样本..."
                        progress_bar = st.progress(0)
                        
                        # 生成测试样本
                        with st.spinner(progress_text):
                            progress_bar.progress(25)
                            samples = generate_samples(task_description)
                            
                            if samples:
                                progress_bar.progress(50)
                                st.session_state.samples = samples
                                
                                # 执行初始提示
                                progress_bar.progress(75)
                                outputs = run_current_best_prompt()
                                
                                if outputs:
                                    progress_bar.progress(100)
                                    time.sleep(0.5)  # 给用户一点时间看到100%
                                    st.success("✅ 初始化完成！正在进入优化过程...")
                                    st.session_state.initialized = True
                                    st.session_state.current_view = "optimization"
                                    st.session_state.is_optimizing = True
                                    time.sleep(1)  # 给用户一点时间阅读成功消息
                                    st.rerun()
                            else:
                                progress_bar.progress(100)
                                st.error("❌ 生成测试样本失败")
                    else:
                        st.error("❌ API配置失败")
                except Exception as e:
                    st.error(f"❌ 初始化失败: {str(e)}")
    
    # 显示一些使用提示
    st.markdown("<h2 class='sub-header'>使用指南</h2>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.expander("📘 SPO+是什么？", expanded=False):
        st.markdown("""
        SPO+ (Self-Prompting Optimization Plus) 是一个增强型自监督提示优化系统，可以自动优化您的AI提示词。
        
        系统通过以下步骤工作：
        1. 分析您的任务需求和初始提示词
        2. 自动生成测试样本
        3. 执行优化迭代，生成更好的提示词
        4. 评估并比较不同提示词的效果
        5. 提供详细的分析和优化结果
        """)
    
    with st.expander("🔍 如何获得最佳效果？", expanded=False):
        st.markdown("""
        为了获得最佳优化效果，建议您：
        
        - **提供详细的任务描述**：清晰说明您希望AI完成的任务、目标受众和期望的输出风格
        - **提供有质量的初始提示词**：初始提示词越好，最终优化结果也会越好
        - **适当设置迭代次数**：简单任务5-8次迭代足够，复杂任务可能需要10-15次
        - **使用自动模式**：系统会自动完成整个优化过程
        """)
        
    with st.expander("🔑 如何获取API密钥？", expanded=False):
        st.markdown("""
        您需要SiliconCloud API密钥才能使用本系统：
        
        1. 访问 [SiliconCloud官网](https://cloud.siliconflow.cn/)
        2. 注册并登录您的账户
        3. 导航到API密钥页面创建新密钥
        4. 复制生成的密钥并粘贴到上方的API Key输入框
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    # 如果加载示例
    if 'example_loaded' in st.session_state and st.session_state.example_loaded:
        st.session_state.example_loaded = False
        st.session_state._example_task = "创建一个AI助手，能够回答用户关于健康和营养的问题，提供科学准确且易于理解的信息。"
        st.session_state._example_prompt = "我是一个健康顾问AI助手。请告诉我你的问题，我会尽力提供帮助。"
        # 使用JavaScript自动填充表单
        st.markdown(
            f"""
            <script>
                setTimeout(function() {{
                    document.querySelector('textarea[aria-label="任务需求描述"]').value = "{st.session_state._example_task}";
                    document.querySelector('textarea[aria-label="初始提示词"]').value = "{st.session_state._example_prompt}";
                    
                    // 触发textArea的input事件以更新Streamlit的状态
                    const taskEvent = new Event('input', {{ bubbles: true }});
                    document.querySelector('textarea[aria-label="任务需求描述"]').dispatchEvent(taskEvent);
                    
                    const promptEvent = new Event('input', {{ bubbles: true }});
                    document.querySelector('textarea[aria-label="初始提示词"]').dispatchEvent(promptEvent);
                }}, 500);
            </script>
            """,
            unsafe_allow_html=True
        )

# 优化视图
def show_optimization_view():
    st.markdown("<h1 class='main-header'>SPO+ 优化过程</h1>", unsafe_allow_html=True)
    
    # 步骤指示器
    st.markdown("""
    <div class="step-container">
        <div class="step completed">
            1
            <div class="step-label">API配置</div>
        </div>
        <div class="step completed">
            2
            <div class="step-label">任务设置</div>
        </div>
        <div class="step active">
            3
            <div class="step-label">优化过程</div>
        </div>
        <div class="step">
            4
            <div class="step-label">优化结果</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 状态栏
    progress_value = st.session_state.current_iteration / st.session_state.max_iterations
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 6, 2])
    
    with col1:
        st.markdown(f"<h4>迭代进度: {st.session_state.current_iteration}/{st.session_state.max_iterations}</h4>", unsafe_allow_html=True)
    
    with col2:
        progress = st.progress(progress_value)
    
    with col3:
        if st.session_state.is_optimizing:
            status = "🔄 正在优化..."
        else:
            status = "⏸️ 等待操作..."
        st.markdown(f"<h4>状态: {status}</h4>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 如果是第一次迭代，显示引导提示
    if st.session_state.current_iteration == 0:
        st.markdown("""
        <div class="guide-box">
            <h4>🚀 优化过程已启动</h4>
            <p>系统正在优化您的提示词。每次迭代包括：生成新提示词 → 测试效果 → 评估结果 → 决定是否保留。您可以在下方查看实时进展。</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 主要内容
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h2 class='sub-header'>提示词比较</h2>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**📝 当前最佳提示**")
        st.text_area("current_best_prompt", value=st.session_state.current_best_prompt, height=200, label_visibility="collapsed")
        
        if st.session_state.new_prompt:
            st.markdown("**✨ 新候选提示**")
            st.text_area("new_prompt", value=st.session_state.new_prompt, height=200, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis:
            st.markdown("<h2 class='sub-header'>改进分析</h2>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(st.session_state.analysis)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 控制按钮
        if not st.session_state.is_optimizing:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ 继续优化", key="continue_btn", use_container_width=True):
                    st.session_state.is_optimizing = True
                    st.rerun()
            
            with col2:
                if st.button("✅ 完成优化", key="finish_btn", use_container_width=True):
                    st.session_state.current_view = "results"
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h2 class='sub-header'>测试样本</h2>", unsafe_allow_html=True)
        
        # 显示样本和输出
        for sample in st.session_state.samples:
            sample_id = sample['id']
            
            with st.expander(f"样本 {sample_id}: {sample['question'][:50]}...", expanded=(sample_id == 1)):
                st.markdown(f"**问题:**")
                st.markdown(f'<div class="output-container">{sample["question"]}</div>', unsafe_allow_html=True)
                
                if sample_id in st.session_state.current_best_outputs:
                    st.markdown("**当前输出:**")
                    current_output = st.session_state.current_best_outputs[sample_id]
                    st.markdown(f"<div class='output-container'>{current_output}</div>", unsafe_allow_html=True)
                
                if st.session_state.new_outputs and sample_id in st.session_state.new_outputs:
                    st.markdown("**新输出:**")
                    new_output = st.session_state.new_outputs[sample_id]
                    
                    # 添加评估结果样式
                    css_class = ""
                    result_icon = ""
                    if st.session_state.evaluations and sample_id in st.session_state.evaluations:
                        result = st.session_state.evaluations[sample_id]
                        if result == "B更好":
                            css_class = "better"
                            result_icon = "✅ "
                        elif result == "A更好":
                            css_class = "worse"
                            result_icon = "❌ "
                        else:
                            css_class = "similar"
                            result_icon = "⚖️ "
                    
                    st.markdown(f"<div class='output-container {css_class}'>{new_output}</div>", unsafe_allow_html=True)
                    
                    if st.session_state.evaluations and sample_id in st.session_state.evaluations:
                        result = st.session_state.evaluations[sample_id]
                        st.markdown(f"**评估结果:** {result_icon}{result}")
        
        # 优化历史
        if st.session_state.optimization_history:
            st.markdown("<h2 class='sub-header'>优化历史</h2>", unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for item in st.session_state.optimization_history:
                css_class = "better" if item["is_better"] else "not-better"
                icon = "✅" if item["is_better"] else "⚠️"
                
                with st.expander(f"{icon} 迭代 {item['iteration']} - {'改进成功' if item['is_better'] else '未改进'}", expanded=False):
                    st.markdown(f"**提示词:**")
                    st.text_area(f"prompt_{item['iteration']}", value=item['prompt'], height=100, label_visibility="collapsed")
                    
                    st.markdown("**评估结果:**")
                    for sample_id, result in item['evaluations'].items():
                        if result == "B更好":
                            result_emoji = "✅"
                        elif result == "A更好":
                            result_emoji = "❌"
                        else:
                            result_emoji = "⚖️"
                        st.markdown(f"- 样本 {sample_id}: {result_emoji} {result}")
                    
                    with st.expander("📊 详细分析", expanded=False):
                        st.markdown(item['analysis'])
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 如果正在优化且需要执行下一步
    if st.session_state.is_optimizing:
        run_optimization_step_with_ui()

# 带UI反馈的优化步骤执行
def run_optimization_step_with_ui():
    if st.session_state.current_iteration >= st.session_state.max_iterations:
        st.session_state.is_optimizing = False
        st.session_state.current_view = "results"
        st.rerun()
        return
    
    st.session_state.current_iteration += 1
    
    # 创建容器来显示进度
    status_container = st.empty()
    output_container = st.empty()
    
    status_container.markdown(f"### 🔄 执行第 {st.session_state.current_iteration} 次优化...")
    
    # 1. 生成新提示候选
    output_container.markdown("🧠 正在生成优化后的提示词...")
    new_prompt = optimize_prompt(
        st.session_state.current_best_prompt,
        json.dumps(st.session_state.current_best_outputs),
        st.session_state.task_description,
        get_optimization_history_summary()
    )
    
    if not new_prompt:
        output_container.error("❌ 优化提示词失败")
        st.session_state.is_optimizing = False
        return
    
    st.session_state.new_prompt = new_prompt
    output_container.markdown("✅ 提示词生成完成")
    
    # 2. 执行新提示
    output_container.markdown("🔍 正在测试新提示词...")
    
    # 检查是否使用流式输出
    use_stream = st.session_state.get('use_streaming', False)
    
    new_outputs = {}
    # 为每个样本创建一个进度条
    progress_bars = {}
    output_blocks = {}
    
    for sample in st.session_state.samples:
        sample_id = sample['id']
        progress_bars[sample_id] = output_container.progress(0)
        output_blocks[sample_id] = output_container.empty()
        
        # 更新状态
        status_container.markdown(f"### 🔄 测试样本 {sample_id}/{len(st.session_state.samples)}...")
        
        if use_stream:
            # 使用流式输出
            output_blocks[sample_id].markdown(f"生成样本 {sample_id} 的回答:")
            output = execute_prompt(new_prompt, sample['question'], use_stream=True)
        else:
            # 常规输出
            output = execute_prompt(new_prompt, sample['question'])
            # 模拟进度
            for i in range(10):
                progress_bars[sample_id].progress((i+1)/10)
                time.sleep(0.1)
            output_blocks[sample_id].markdown(f"样本 {sample_id} 回答已完成")
            
        new_outputs[sample_id] = output
        progress_bars[sample_id].progress(1.0)
    
    # 清理进度条和块
    for sample_id in progress_bars:
        progress_bars[sample_id].empty()
        output_blocks[sample_id].empty()
        
    st.session_state.new_outputs = new_outputs
    output_container.markdown("✅ 测试完成")
    
    # 3. 评估新旧输出
    output_container.markdown("⚖️ 正在评估输出质量...")
    evaluations = {}
    
    eval_progress = output_container.progress(0)
    for idx, sample in enumerate(st.session_state.samples):
        sample_id = sample['id']
        output_a = st.session_state.current_best_outputs.get(sample_id, "")
        output_b = new_outputs.get(sample_id, "")
        
        evaluation = evaluate_outputs(
            output_a,
            output_b,
            st.session_state.task_description,
            sample['question']
        )
        
        evaluations[sample_id] = evaluation
        eval_progress.progress((idx + 1) / len(st.session_state.samples))
    
    eval_progress.empty()
    st.session_state.evaluations = evaluations
    output_container.markdown("✅ 评估完成")
    
    # 4. 分析提示变化
    output_container.markdown("🔎 正在分析提示词变化...")
    analysis = analyze_changes(
        st.session_state.current_best_prompt,
        new_prompt,
        st.session_state.task_description
    )
    
    st.session_state.analysis = analysis
    output_container.markdown("✅ 分析完成")
    
    # 5. 根据评估结果更新最佳提示
    is_better = should_update_best_prompt(evaluations)
    if is_better:
        st.session_state.current_best_prompt = new_prompt
        st.session_state.current_best_outputs = new_outputs
        output_container.markdown("🎉 发现更好的提示词！已更新为当前最佳提示。")
    else:
        output_container.markdown("📌 新提示词未能提供改进，保持当前最佳提示不变。")
    
    # 记录优化历史
    st.session_state.optimization_history.append({
        "iteration": st.session_state.current_iteration,
        "prompt": new_prompt,
        "is_better": is_better,
        "analysis": analysis,
        "evaluations": evaluations
    })
    
    # 清空状态容器和输出容器
    status_container.empty()
    output_container.empty()
    
    # 如果是自动模式，继续优化
    if st.session_state.auto_mode:
        time.sleep(1)  # 短暂延迟，让UI更新
        st.rerun()  # 重新加载页面继续优化
    else:
        st.session_state.is_optimizing = False
        st.rerun()  # 重新加载页面等待用户操作

# 结果视图
def show_results_view():
    st.markdown("<h1 class='main-header'>SPO+ 优化结果</h1>", unsafe_allow_html=True)
    
    # 步骤指示器
    st.markdown("""
    <div class="step-container">
        <div class="step completed">
            1
            <div class="step-label">API配置</div>
        </div>
        <div class="step completed">
            2
            <div class="step-label">任务设置</div>
        </div>
        <div class="step completed">
            3
            <div class="step-label">优化过程</div>
        </div>
        <div class="step active">
            4
            <div class="step-label">优化结果</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 结果摘要
    st.markdown("<h2 class='sub-header'>优化摘要</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("总迭代次数", st.session_state.current_iteration)
    
    with col2:
        better_count = sum(1 for item in st.session_state.optimization_history if item["is_better"])
        st.metric("成功改进次数", better_count)
    
    with col3:
        improvement_rate = better_count / st.session_state.current_iteration if st.session_state.current_iteration > 0 else 0
        st.metric("改进成功率", f"{improvement_rate:.0%}")
    
    # 提示消息
    if improvement_rate > 0.5:
        st.success("🎉 优化效果良好！超过半数的迭代提供了改进。")
    elif improvement_rate > 0.2:
        st.info("👍 优化取得了一定成效，但仍有改进空间。")
    else:
        st.warning("⚠️ 优化过程较为困难，可能需要更好的初始提示词或更精确的任务描述。")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 最终提示词
    st.markdown("<h2 class='sub-header'>最终优化提示词</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    final_prompt = st.text_area("final_prompt", value=st.session_state.current_best_prompt, height=300, label_visibility="collapsed")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="tip-box">
            <p>💡 提示：您可以将此提示词用于实际应用。改进后的提示词应该能够生成更高质量的AI回复。</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        copy_button = st.button("📋 复制提示词", use_container_width=True)
        
    if copy_button:
        st.code(st.session_state.current_best_prompt, language="markdown")
        st.success("✅ 提示词已生成代码块，可直接复制！")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 优化效果测试
    st.markdown("<h2 class='sub-header'>优化效果测试</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="guide-box">
        <h4>🧪 测试优化后的提示词</h4>
        <p>输入一个问题，看看优化后的提示词如何回答。您可以打开流式输出查看生成过程。</p>
    </div>
    """, unsafe_allow_html=True)
    
    user_question = st.text_area("输入测试问题", placeholder="请输入一个问题来测试优化后的提示词效果...", height=100)
    use_stream_test = st.checkbox("使用流式输出", value=True)
    
    if st.button("🚀 测试提示词", use_container_width=True):
        if user_question:
            with st.spinner("正在生成回答..."):
                response = execute_prompt(
                    st.session_state.current_best_prompt, 
                    user_question,
                    use_stream=use_stream_test
                )
            if not use_stream_test:
                st.markdown("<div class='output-container better'>", unsafe_allow_html=True)
                st.markdown(response)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ 请输入测试问题")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 优化历史
    if st.session_state.optimization_history:
        st.markdown("<h2 class='sub-header'>优化历史</h2>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # 图表显示优化进度
        iteration_data = [(i+1) for i in range(len(st.session_state.optimization_history))]
        better_data = [1 if item["is_better"] else 0 for item in st.session_state.optimization_history]
        
        import pandas as pd
        import altair as alt
        
        df = pd.DataFrame({
            '迭代': iteration_data,
            '是否改进': better_data
        })
        
        base = alt.Chart(df).encode(
            x=alt.X('迭代:O', axis=alt.Axis(title='迭代次数')),
            y=alt.Y('是否改进:Q', axis=alt.Axis(title='改进状态'))
        )
        
        bars = base.mark_bar(color='#3B82F6').encode(
            color=alt.condition(
                alt.datum['是否改进'] == 1,
                alt.value('#10B981'),  # 成功改进
                alt.value('#EF4444')   # 未改进
            )
        )
        
        st.altair_chart(bars, use_container_width=True)
        
        # 优化历史详情
        for item in st.session_state.optimization_history:
            css_class = "better" if item["is_better"] else "not-better"
            icon = "✅" if item["is_better"] else "⚠️"
            
            with st.expander(f"{icon} 迭代 {item['iteration']} - {'改进成功' if item['is_better'] else '未改进'}", expanded=False):
                st.markdown(f"**提示词:**")
                st.text_area(f"result_prompt_{item['iteration']}", value=item['prompt'], height=100, label_visibility="collapsed")
                
                better_count = sum(1 for result in item['evaluations'].values() if result == "B更好")
                worse_count = sum(1 for result in item['evaluations'].values() if result == "A更好")
                similar_count = sum(1 for result in item['evaluations'].values() if result == "相似")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("更好结果", better_count)
                with col2:
                    st.metric("更差结果", worse_count)
                with col3:
                    st.metric("相似结果", similar_count)
                
                st.markdown("**评估结果详情:**")
                for sample_id, result in item['evaluations'].items():
                    if result == "B更好":
                        result_emoji = "✅"
                        result_color = "#10B981"
                    elif result == "A更好":
                        result_emoji = "❌"
                        result_color = "#EF4444"
                    else:
                        result_emoji = "⚖️"
                        result_color = "#F59E0B"
                    st.markdown(f"<span style='color:{result_color}'>- 样本 {sample_id}: {result_emoji} {result}</span>", unsafe_allow_html=True)
                
                with st.expander("📊 详细分析", expanded=False):
                    st.markdown(item['analysis'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 导出和重置按钮
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 导出历史", use_container_width=True):
            export_data = {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task_description": st.session_state.task_description,
                "initial_prompt": st.session_state.current_best_prompt,
                "iterations": st.session_state.current_iteration,
                "history": [{
                    "iteration": item["iteration"],
                    "prompt": item["prompt"],
                    "is_better": item["is_better"],
                    "analysis": item["analysis"]
                } for item in st.session_state.optimization_history]
            }
            
            st.download_button(
                label="📥 下载JSON文件",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"spo-plus-history-{time.strftime('%Y%m%d-%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        if st.button("🔄 开始新的优化", use_container_width=True):
            # 保存API设置以便重用
            api_key = st.session_state.get('api_key', '')
            base_url = st.session_state.get('base_url', API_BASE_URL)
            
            # 重置会话状态
            for key in list(st.session_state.keys()):
                if key != "available_models":
                    del st.session_state[key]
            
            # 恢复API设置
            st.session_state.api_key = api_key
            st.session_state.base_url = base_url
            
            # 重置状态
            st.session_state.initialized = False
            st.session_state.api_configured = api_key != ''
            st.session_state.current_view = "config"
            st.session_state.current_iteration = 0
            st.session_state.max_iterations = 10
            st.session_state.samples = []
            st.session_state.current_best_prompt = ""
            st.session_state.current_best_outputs = {}
            st.session_state.new_prompt = ""
            st.session_state.new_outputs = {}
            st.session_state.evaluations = {}
            st.session_state.analysis = ""
            st.session_state.optimization_history = []
            st.session_state.is_optimizing = False
            
            st.success("✅ 已重置！您可以开始新的优化任务。")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# 主应用
def main():
    # 侧边栏
    with st.sidebar:
        st.markdown("")
        st.markdown("增强型自监督提示优化系统")
        
        st.markdown("---")
        
        if st.button("配置"):
            st.session_state.current_view = "config"
            st.rerun()
        
        if st.session_state.initialized:
            if st.button("优化过程"):
                st.session_state.current_view = "optimization"
                st.rerun()
            
            if st.button("优化结果"):
                st.session_state.current_view = "results"
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 关于")
        st.markdown("""
        SPO+是一个强大的提示词优化系统，结合了自动化优化和人类反馈，帮助用户创建更高质量的提示词。
        
        **特点:**
        - 自动优化
        - 流式输出
        - 人机协作
        - 可视化界面
        """)
    
    # 主内容
    if st.session_state.current_view == "config":
        show_config_view()
    elif st.session_state.current_view == "optimization":
        show_optimization_view()
    elif st.session_state.current_view == "results":
        show_results_view()

if __name__ == "__main__":
    main() 
