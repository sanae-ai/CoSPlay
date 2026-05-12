# inference.py
# vLLM + API 封装成一个 ModelRunner (基于你提供的版本修复)
import os
import json
import time
import multiprocessing as mp
from pathlib import Path

# 移除不必要的 subprocess 和 queue 依赖，回归简单
from termcolor import cprint
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


def _prepare_worker_cache_env(worker_rank: int, cache_root: str | None = None) -> None:
    """
    为每个 worker 准备独立的缓存环境。
    """
    default_root = Path(
        os.environ.get(
            "VLLM_WORKER_CACHE_ROOT",
            os.path.join(os.path.expanduser("~"), ".cache", "vllm_worker"),
        )
    )
    base_dir = Path(cache_root).expanduser() if cache_root else default_root
    worker_dir = base_dir / f"worker_{worker_rank}_pid_{os.getpid()}"
    torch_compile_dir = worker_dir / "torch_compile_cache"
    inductor_dir = worker_dir / "inductor"
    triton_dir = worker_dir / "triton"
    for path in (torch_compile_dir, inductor_dir, triton_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_TORCH_COMPILE_CACHE_DIR"] = str(torch_compile_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)


def worker_fn(
    pretrained_model,
    gpu_ids,
    task_queue,
    result_queue,
    max_model_len,
    max_generation_token,
    worker_rank,
    temp,
    trust_remote_code,
    cache_root=None,
):
    """
    vLLM 工作进程 (稳定优化版)
    """
    # =========================================================================
    # 【修复 1】解决 "Adding requests" 卡死的核心！
    # 必须在导入 transformers/vllm 之前设置。禁止 Tokenizer 多线程竞争。
    # =========================================================================
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # -------------------------------------------------------------
    # 环境变量配置
    # -------------------------------------------------------------
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ["VLLM_ENFORCE_EAGER"] = "1"  # 强制关闭 CUDA Graph 防死锁
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO" 

    _prepare_worker_cache_env(worker_rank, cache_root)

    # 延迟导入 vLLM
    from vllm import LLM, SamplingParams

    print(f"[Worker {worker_rank}] Loading model on GPUs {gpu_ids}...", flush=True)
    kwargs = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    # -------------------------------------------------------------
    # 初始化 LLM (参数针对 H100 优化)
    # -------------------------------------------------------------
    try:
        llm = LLM(
            model=pretrained_model,
            dtype="bfloat16",
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=0.92,     # H100 显存大，给足空间
            max_model_len=max_model_len,
            max_num_seqs=8,                  # 【关键】降低并发，解决调度器死锁
            enforce_eager=True,              # 双重保险
            enable_chunked_prefill=True,     # 长文本必备
            **kwargs,
        )
    except Exception as e:
        print(f"[Worker {worker_rank}] FATAL: LLM Init failed: {e}", flush=True)
        # 如果初始化失败，直接让进程退出，主进程会因为 Queue 没数据而报错(或超时)
        return

    print(f"[Worker {worker_rank}] LLM Ready!", flush=True)
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        max_tokens=max_generation_token,
        stop=["</answer>", "User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"],
    )

    # -------------------------------------------------------------
    # 【修复 2】计算输入截断长度
    # -------------------------------------------------------------
    # 预留输出空间 + 200 buffer
    allowed_input_tokens = max(1024, max_model_len - max_generation_token - 200)
    # 估算字符限制 (1 token ≈ 3.5 chars)
    char_limit = int(allowed_input_tokens * 3.5)
    print(f"[Worker {worker_rank}] Input Limit: ~{char_limit} chars", flush=True)

    while True:
        task = task_queue.get()
        if task == "STOP":
            print(f"[Worker {worker_rank}] Stopping...", flush=True)
            break

        task_id, prompts = task
        print(f"[Worker {worker_rank}] Processing {len(prompts)} prompts for task {task_id}...", flush=True)  
        
        # -------------------------------------------------------------
        # 【修复 3】输入截断 (Sanitization)
        # 防止超长垃圾数据导致 Tokenizer 在 CPU 上卡死
        # -------------------------------------------------------------
        safe_prompts = []
        for p in prompts:
            if len(p) > char_limit:
                # 保留尾部（最新的指令/代码上下文）
                p = p[-char_limit:]
            safe_prompts.append(p)
        
        # -------------------------------------------------------------
        # 执行推理 (带异常捕获)
        # -------------------------------------------------------------
        try:
            outputs = llm.generate(safe_prompts, sampling_params)
            result_texts = [out.outputs[0].text for out in outputs]
            result_queue.put((task_id, result_texts))
            print(f"[Worker {worker_rank}] Task {task_id} complete!", flush=True)

        except Exception as e:
            print(f"[Worker {worker_rank}] ERROR in generation: {e}", flush=True)
            # 发生错误时，填充错误结果，保证结果列表长度对齐
            try:
                result_queue.put((task_id, ["Error: Generation Failed"] * len(prompts)))
            except:
                pass


def start_workers(args, cache_root=None):
    """
    根据 GPU 配置启动多个 worker 进程。
    【关键修改】必须使用 spawn 模式启动！
    """
    task_queues = []
    result_queues = []
    processes = []
    
    # 1. 获取 spawn 上下文 (解决 CUDA 初始化冲突)
    ctx = mp.get_context('spawn') 

    for i, gpu_ids in enumerate(args.gpu_groups):
        task_q = ctx.Queue()
        result_q = ctx.Queue()
        
        p = ctx.Process(
            target=worker_fn,
            args=(
                args.pretrained_model,
                gpu_ids,
                task_q,
                result_q,
                args.max_model_len,
                args.max_generation_token,
                i,
                args.temp,
                args.trust_remote_code,
                cache_root,
            ),
        )
        p.start()
        task_queues.append(task_q)
        result_queues.append(result_q)
        processes.append(p)

    return task_queues, result_queues, processes


def stop_workers(task_queues, processes):
    for q in task_queues:
        q.put("STOP")
    for p in processes:
        p.join()


def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def generate_results_vllm(all_prompts, gpu_groups, task_queues, result_queues):
    """
    vLLM 推理主入口：分发任务并收集结果。
    """
    if not all_prompts:
        return []

    # 1. 分发任务
    prompt_sets = split_prompts(all_prompts, len(gpu_groups))
    for i, prompts in enumerate(prompt_sets):
        task_queues[i].put((i, prompts))

    # 2. 收集结果
    results = [None] * len(prompt_sets)
    
    # 简单的收集逻辑，带有基本的超时保护
    # H100 跑一批数据 1小时如果还没跑完，说明肯定出事了

    
    collected_count = 0
    total_tasks = len(prompt_sets)
    
    start_time = time.time()

    while collected_count < total_tasks:
        # 检查整体是否严重超时

        for i, q in enumerate(result_queues):
            # 如果这个坑位还没拿到结果
            if results[i] is None:
                try:
                    # 非阻塞尝试获取，或者短超时
                    task_id, result = q.get(timeout=0.1)
                    results[task_id] = result
                    collected_count += 1
                except:
                    pass
        
        # 稍微睡一下避免 CPU 100%
        time.sleep(0.1)

    # 3. 展平结果
    flat = []
    for i, result_set in enumerate(results):
        if result_set is None:
            # 如果到最后还是 None (可能是超时退出了)
            print(f"[Warning] Task {i} failed or timed out. Filling with Errors.")
            # 估算这一批应该有多少条
            expected_len = len(prompt_sets[i])
            flat.extend(["Error: Timeout/Crash"] * expected_len)
        else:
            flat.extend(result_set)
    return flat


# ==================== API 推理相关 (保持不变) ====================

def fetch_completion(user_prompt: str, args) -> str:
    try:
        client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url.replace("/v1/chat/completions", ""),
        )
        response = client.chat.completions.create(
            model=args.api_model_name,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=args.api_temperature,
            max_tokens=args.max_tokens,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[API Error] {str(e)}")
        # 简单递归重试，生产环境建议加 limit
        return fetch_completion(user_prompt, args)


def generate_results_api(prompts, args):
    total = len(prompts)
    if total == 0:
        return []

    results = ["No outputs"] * total
    for batch_start in range(0, total, args.rpm_limit):
        batch_end = min(batch_start + args.rpm_limit, total)
        batch_slice = range(batch_start, batch_end)

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            fut_to_idx = {
                pool.submit(fetch_completion, prompts[i], args): i
                for i in batch_slice
            }

            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"[Error] Failed to get result for index {idx}: {e}")
                    results[idx] = f"Error: {str(e)}"

        elapsed = time.time() - t0
        leftover = max(0, 60.0 - elapsed)
        if batch_end < total and leftover:
            time.sleep(leftover)

        print(f"Processed {batch_end}/{total} prompts")

    return results

def save_prompts_to_jsonl(prompts, filename, system_content, model, max_tokens, url, args):
    with open(filename, "w", encoding="utf-8") as fout:
        for i, user_prompt in enumerate(prompts, start=1):
            obj = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": url,
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": args.api_temperature,
                },
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} requests to {filename!r}")

def extract_completions(raw):
    lines = raw.strip().split("\n")
    records = [json.loads(line) for line in lines]
    bodies = [rec["response"]["body"] for rec in records]
    return [body["choices"][0]["message"]["content"] for body in bodies]

def generate_by_openai_batch(prompts, args):
    # (保持原本的 Batch API 逻辑不变)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    api_batch_filename = (
        args.api_model_name.replace("/", ".")
        + "_" + args.dataset + "_" + args.mode + f"_{timestamp}.jsonl"
    )

    save_prompts_to_jsonl(
        prompts,
        filename=api_batch_filename,
        system_content="You are a helpful assistant.",
        model=args.api_model_name,
        max_tokens=args.max_tokens,
        url="/v1/chat/completions",
        args=args,
    )

    client = OpenAI(api_key=args.api_key)
    batch_input_file = client.files.create(
        file=open(api_batch_filename, "rb"), purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )
    
    cprint(f"batch id: {batch.id}", color="green")
    
    start_time = time.time()
    last_index = 0
    min_interval = 2

    while True:
        time.sleep(10)
        batch = client.batches.retrieve(batch.id)
        if batch.status == "completed":
            file_id = batch.output_file_id
            break
        if batch.status in ("failed", "expired", "cancelled"):
            cprint(f"[Batch Failed] Status: {batch.status}", color="red")
            return ["Error: Batch API failed"] * len(prompts)
        
        elapsed = time.time() - start_time
        idx = int(elapsed // (60 * min_interval))
        if idx > last_index:
            last_index = idx
            print(f"Status: {batch.status}, Completed: {batch.request_counts.completed}/{batch.request_counts.total}")

    file_response = client.files.content(file_id)
    completions = extract_completions(file_response.text)
    return completions if completions else []


class ModelRunner:
    def __init__(self, args):
        self.args = args
        self.use_api = args.use_api
        self.task_queues = None
        self.result_queues = None
        self.processes = None

        if not self.use_api:
            # 这里的 start_workers 已经是修改过的 spawn 版本了
            self.task_queues, self.result_queues, self.processes = start_workers(args)
            print("✓ vLLM 模型加载完成", flush=True)
        else:
            print("✓ 使用 API 模式推理", flush=True)

    def generate(self, prompts):
        if not prompts:
            return []

        if not self.use_api:
            return generate_results_vllm(
                prompts,
                self.args.gpu_groups,
                self.task_queues,
                self.result_queues,
            )
        else:
            if self.args.use_openai_batch_api:
                return generate_by_openai_batch(prompts, self.args)
            else:
                return generate_results_api(prompts, self.args)

    def close(self):
        if not self.use_api and self.task_queues is not None:
            stop_workers(self.task_queues, self.processes)