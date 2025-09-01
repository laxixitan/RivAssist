import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, simpledialog, messagebox
import sys
import os
import json
import threading
import time
import subprocess
import queue
import re
import traceback
import shutil
from datetime import datetime
import tempfile
from tkinter import messagebox, ttk

# 全局变量：防止环境检查递归
ENVIRONMENT_CHECK_IN_PROGRESS = False
ENVIRONMENT_CHECK_LOCK_FILE = "env_check.lock"

# 国内PyPI镜像源
PYPI_MIRRORS = {
    "阿里云": "https://mirrors.aliyun.com/pypi/simple/",
    "豆瓣": "https://pypi.doubanio.com/simple/",
    "清华大学": "https://pypi.tuna.tsinghua.edu.cn/simple/",
    "中国科学技术大学": "https://pypi.mirrors.ustc.edu.cn/simple/",
    "官方源": "https://pypi.org/simple/"  # 新增官方源
}

# 必要的依赖包
REQUIRED_PACKAGES = {
    "requests": "requests",
    "matplotlib": "matplotlib",
    "pillow": "Pillow"
}

# 检查是否是打包后的EXE环境
def is_standalone_exe():
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

# 如果是EXE模式，隐藏控制台窗口
if is_standalone_exe():
    try:
        import ctypes
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    except:
        pass

def create_environment_check_lock():
    """创建环境检查锁，防止多个实例同时进行环境检查"""
    global ENVIRONMENT_CHECK_IN_PROGRESS
    
    # 检查是否已经有锁文件
    if os.path.exists(ENVIRONMENT_CHECK_LOCK_FILE):
        # 检查锁文件是否超过10分钟，如果是则视为过期并删除
        try:
            file_mtime = os.path.getmtime(ENVIRONMENT_CHECK_LOCK_FILE)
            if time.time() - file_mtime > 600:  # 10分钟
                os.remove(ENVIRONMENT_CHECK_LOCK_FILE)
        except:
            pass
    
    try:
        # 尝试创建并锁定文件
        lock_file = open(ENVIRONMENT_CHECK_LOCK_FILE, 'w')
        if os.name == 'nt':
            # Windows系统使用msvcrt锁定文件
            import msvcrt
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            # Unix系统使用fcntl锁定文件
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except:
                pass
                
        ENVIRONMENT_CHECK_IN_PROGRESS = True
        return lock_file
    except:
        # 无法获取锁，说明已有环境检查在进行
        ENVIRONMENT_CHECK_IN_PROGRESS = True
        return None

def release_environment_check_lock(lock_file):
    """释放环境检查锁"""
    global ENVIRONMENT_CHECK_IN_PROGRESS
    try:
        if lock_file:
            # 在Windows上解锁文件
            if os.name == 'nt':
                try:
                    import msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except:
                    pass
            lock_file.close()
        if os.path.exists(ENVIRONMENT_CHECK_LOCK_FILE):
            os.remove(ENVIRONMENT_CHECK_LOCK_FILE)
    except:
        pass
    ENVIRONMENT_CHECK_IN_PROGRESS = False

class DependencyManager:
    """依赖管理类，用于检测和安装缺失的Python模块"""

    def __init__(self, parent_window, python_path=None):
        self.parent = parent_window  # 父窗口引用，用于UI操作
        self.python_path = python_path or self.find_python_interpreter()
        self.install_queue = queue.Queue()  # 安装任务队列
        self.is_installing = False  # 安装状态标记
        self.output_queue = queue.Queue()  # 输出消息队列
        self.mirror_urls = {
            "阿里云": "https://mirrors.aliyun.com/pypi/simple/",
            "豆瓣": "https://pypi.doubanio.com/simple/",
            "清华大学": "https://pypi.tuna.tsinghua.edu.cn/simple/",
            "中国科学技术大学": "https://pypi.mirrors.ustc.edu.cn/simple/",
            "官方源": "https://pypi.org/simple/",
        }
        self.current_mirror = "阿里云"  # 默认镜像源
        self.installation_window = None  # 安装进度窗口
        # 关键修改：模块别名映射表（导入名 → 实际安装包名）
        self.module_aliases = {
            "PIL": "Pillow",
            "cv2": "opencv-python",
            "yaml": "PyYAML",
            "bs4": "beautifulsoup4",
            "sklearn": "scikit-learn",
            "tensorflow": "tensorflow",
            "keras": "keras",
            "torch": "torch",
            "pytorch": "torch",
            "mxnet": "mxnet",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "plotly": "plotly",
            "seaborn": "seaborn",
            "pandas": "pandas",
            "numpy": "numpy",
            "scipy": "scipy",
            "django": "django",
            "flask": "flask",
            "requests": "requests",
            "urllib3": "urllib3",
            "selenium": "selenium",
            "lxml": "lxml",
            "xlrd": "xlrd",
            "openpyxl": "openpyxl",
            "pytest": "pytest",
            "unittest": "unittest",
            "datetime": "python-dateutil",
            "dateutil": "python-dateutil",
            "six": "six",
            "yaml": "PyYAML",
            "json5": "json5",
            "jsonschema": "jsonschema",
            "regex": "regex",
            "chardet": "chardet",
            "idna": "idna",
            "certifi": "certifi",
            "cryptography": "cryptography",
            "pycryptodome": "pycryptodome",
            "crypto": "pycryptodome",
            "pyjwt": "pyjwt",
            "jwt": "pyjwt",
            "python-dotenv": "python-dotenv",
            "dotenv": "python-dotenv",
            "pygments": "pygments",
            "markdown": "markdown",
            "pygments": "pygments",
            "tqdm": "tqdm",
            "progressbar": "progressbar2",
            "loguru": "loguru",
            "logging": "logging",
            "colorama": "colorama",
            "termcolor": "termcolor",
            "prettytable": "prettytable",
            "tabulate": "tabulate",
            "pymysql": "pymysql",
            "mysql.connector": "mysql-connector-python",
            "psycopg2": "psycopg2-binary",
            "sqlite3": "pysqlite3",
            "redis": "redis",
            "pymongo": "pymongo",
            "mongoengine": "mongoengine",
            "sqlalchemy": "sqlalchemy",
            "alembic": "alembic",
            "pika": "pika",
            "kafka": "kafka-python",
            "boto3": "boto3",
            "botocore": "botocore",
            "azure": "azure",
            "google.cloud": "google-cloud",
            "awscli": "awscli",
            "gcsfs": "gcsfs",
            "s3fs": "s3fs",
            "paramiko": "paramiko",
            "fabric": "fabric",
            "ansible": "ansible",
            "docker": "docker",
            "kubernetes": "kubernetes",
            "pykube": "pykube-ng",
            "gitpython": "gitpython",
            "pygithub": "pygithub",
            "jira": "jira",
            "slack-sdk": "slack-sdk",
            "discord.py": "discord.py",
            "telegram": "python-telegram-bot",
            "whatsapp": "pywhatkit",
            "twilio": "twilio",
            "smtplib": "secure-smtplib",
            "email": "email-validator",
            "pyexcel": "pyexcel",
            "pyexcel-xls": "pyexcel-xls",
            "pyexcel-xlsx": "pyexcel-xlsx",
            "pyexcel-ods": "pyexcel-ods",
            "pdfkit": "pdfkit",
            "pypdf2": "PyPDF2",
            "reportlab": "reportlab",
            "docx": "python-docx",
            "xlwt": "xlwt",
            "xlutils": "xlutils",
            "pandas_excel": "pandas",
            "pandas_datareader": "pandas-datareader",
            "pandas_profiling": "pandas-profiling",
            "numpy_financial": "numpy-financial",
            "scipy_optimize": "scipy",
            "scipy_stats": "scipy",
            "statsmodels": "statsmodels",
            "sympy": "sympy",
            "numba": "numba",
            "cupy": "cupy",
            "jax": "jax",
            "jaxlib": "jaxlib",
            "tensorflow_probability": "tensorflow-probability",
            "torchvision": "torchvision",
            "torchaudio": "torchaudio",
            "transformers": "transformers",
            "datasets": "datasets",
            "tokenizers": "tokenizers",
            "accelerate": "accelerate",
            "diffusers": "diffusers",
            "bitsandbytes": "bitsandbytes",
            "sentence_transformers": "sentence-transformers",
            "faiss": "faiss-cpu",
            "nltk": "nltk",
            "spacy": "spacy",
            "textblob": "textblob",
            "gensim": "gensim",
            "wordcloud": "wordcloud",
            "pyldavis": "pyldavis",
            "langdetect": "langdetect",
            "translate": "translate",
            "deep_translator": "deep-translator",
            "pydub": "pydub",
            "librosa": "librosa",
            "soundfile": "soundfile",
            "opencv-python": "opencv-python",
            "opencv-contrib-python": "opencv-contrib-python",
            "pycocotools": "pycocotools",
            "mmcv": "mmcv-full",
            "mmdet": "mmdet",
            "mmpose": "mmpose",
            "mmsegmentation": "mmsegmentation",
            "detectron2": "detectron2",
            "albumentations": "albumentations",
            "imgaug": "imgaug",
            "pillow": "Pillow",
            "imageio": "imageio",
            "scikit-image": "scikit-image",
            "matplotlib": "matplotlib",
            "plotly": "plotly",
            "pyzbar": "pyzbar",
            "seaborn": "seaborn",
            "bokeh": "bokeh",
            "altair": "altair",
            "holoviews": "holoviews",
            "geopandas": "geopandas",
            "folium": "folium",
            "basemap": "basemap",
            "cartopy": "cartopy",
            "networkx": "networkx",
            "pyvis": "pyvis",
            "igraph": "python-igraph",
            "pygraphviz": "pygraphviz",
            "dask": "dask",
            "ray": "ray",
            "joblib": "joblib",
            "multiprocess": "multiprocess",
            "concurrent": "concurrent-futures",
            "threading": "threading",
            "asyncio": "asyncio",
            "aiohttp": "aiohttp",
            "asyncpg": "asyncpg",
            "aiomysql": "aiomysql",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "starlette": "starlette",
            "django": "django",
            "flask": "flask",
            "tornado": "tornado",
            "sanic": "sanic",
            "bottle": "bottle",
            "pyramid": "pyramid",
            "falcon": "falcon",
            "hug": "hug",
            "quart": "quart",
            "django-rest-framework": "djangorestframework",
            "flask-restful": "flask-restful",
            "flask-restx": "flask-restx",
            "pydantic": "pydantic",
            "marshmallow": "marshmallow",
            "attrs": "attrs",
            "dataclasses": "dataclasses",
            "typing": "typing",
            "mypy": "mypy",
            "pylint": "pylint",
            "flake8": "flake8",
            "black": "black",
            "isort": "isort",
            "autoflake": "autoflake",
            "yapf": "yapf",
            "docformatter": "docformatter",
            "pydocstyle": "pydocstyle",
            "bandit": "bandit",
            "safety": "safety",
            "pip-audit": "pip-audit",
            "pytest": "pytest",
            "pytest-cov": "pytest-cov",
            "pytest-mock": "pytest-mock",
            "tox": "tox",
            "nox": "nox",
            "coverage": "coverage",
            "unittest": "unittest",
            "doctest": "doctest",
            "hypothesis": "hypothesis",
            "freezegun": "freezegun",
            "pytest-xdist": "pytest-xdist",
            "pytest-parallel": "pytest-parallel",
            "locust": "locust",
            "pytest-benchmark": "pytest-benchmark",
            "memory_profiler": "memory-profiler",
            "line_profiler": "line_profiler",
            "cProfile": "cProfile",
            "py-spy": "py-spy",
            "sentry-sdk": "sentry-sdk",
            "loguru": "loguru",
            "structlog": "structlog",
            "python-json-logger": "python-json-logger",
            "colorlog": "colorlog",
            "rich": "rich",
            "click": "click",
            "fire": "fire",
            "argparse": "argparse",
            "docopt": "docopt",
            "typer": "typer",
            "invoke": "invoke",
            "fabric": "fabric",
            "pycairo": "pycairo",
            "pygobject": "pygobject",
            "pyqt5": "pyqt5",
            "pyside2": "pyside2",
            "wxpython": "wxpython",
            "kivy": "kivy",
            "pygame": "pygame",
            "arcade": "arcade",
            "pyglet": "pyglet",
            "turtle": "turtle",
            "pyopengl": "pyopengl",
            "moderngl": "moderngl",
            "pyvulkan": "pyvulkan",
            "pymunk": "pymunk",
            "box2d": "box2d-py",
            "pybullet": "pybullet",
            "gym": "gym",
            "gymnasium": "gymnasium",
            "stable-baselines3": "stable-baselines3",
            "ray[rllib]": "ray[rllib]",
            "pettingzoo": "pettingzoo",
            "minigrid": "minigrid",
            "procgen": "procgen",
            "dm-control": "dm-control",
            "mujoco": "mujoco",
            "roboschool": "roboschool",
            "pybulletgym": "pybulletgym",
            "pytorch-lightning": "pytorch-lightning",
            "lightning-bolts": "lightning-bolts",
            "pytorch-ignite": "pytorch-ignite",
            "tensorboard": "tensorboard",
            "wandb": "wandb",
            "mlflow": "mlflow",
            "neptune-client": "neptune-client",
            "comet-ml": "comet-ml",
            "clearml": "clearml",
            "aim": "aim",
            "dvclive": "dvclive",
            "hydra-core": "hydra-core",
            "omegaconf": "omegaconf",
            "pydantic-yaml": "pydantic-yaml",
            "python-dotenv": "python-dotenv",
            "configparser": "configparser",
            "pyyaml": "pyyaml",
            "json5": "json5",
            "toml": "toml",
            "ini": "configparser",
            "xmltodict": "xmltodict",
            "lxml": "lxml",
            "beautifulsoup4": "beautifulsoup4",
            "html5lib": "html5lib",
            "pyquery": "pyquery",
            "cssselect": "cssselect",
            "requests-html": "requests-html",
            "selenium": "selenium",
            "playwright": "playwright",
            "splinter": "splinter",
            "mechanicalsoup": "mechanicalsoup",
            "robobrowser": "robobrowser",
            "scrapy": "scrapy",
            "pyspider": "pyspider",
            "portia": "portia",
            "scrapy-splash": "scrapy-splash",
            "scrapy-playwright": "scrapy-playwright",
            "requests": "requests",
            "httpx": "httpx",
            "aiohttp": "aiohttp",
            "urllib3": "urllib3",
            "httplib2": "httplib2",
            "pycurl": "pycurl",
            "wget": "wget",
            "curlify": "curlify",
            "pysocks": "pysocks",
            "requests[socks]": "requests[socks]",
            "proxymanager": "proxymanager",
            "python-socks": "python-socks",
            "m3u8": "m3u8",
            "ffmpeg-python": "ffmpeg-python",
            "pytube": "pytube",
            "youtube-dl": "youtube-dl",
            "yt-dlp": "yt-dlp",
            "python-vlc": "python-vlc",
            "pydub": "pydub",
            "librosa": "librosa",
            "soundfile": "soundfile",
            "pyaudio": "pyaudio",
            "speechrecognition": "SpeechRecognition",
            "pyttsx3": "pyttsx3",
            "gTTS": "gTTS",
            "deep-speech": "deep-speech",
            "whisper": "openai-whisper",
            "pytesseract": "pytesseract",
            "pyocr": "pyocr",
            "pdf2image": "pdf2image",
            "pikepdf": "pikepdf",
            "pdfrw": "pdfrw",
            "PyPDF2": "PyPDF2",
            "pdfminer": "pdfminer.six",
            "pdfplumber": "pdfplumber",
            "camelot-py": "camelot-py",
            "tabula-py": "tabula-py",
            "pdftotext": "pdftotext",
            "python-magic": "python-magic",
            "filetype": "filetype",
            "python-multipart": "python-multipart",
            "formdata": "formdata",
            "pycryptodome": "pycryptodome",
            "cryptography": "cryptography",
            "pyjwt": "pyjwt",
            "oauthlib": "oauthlib",
            "requests-oauthlib": "requests-oauthlib",
            "python-jose": "python-jose",
            "pyopenssl": "pyopenssl",
            "pyasn1": "pyasn1",
            "pyasn1-modules": "pyasn1-modules",
            "rsa": "rsa",
            "ecdsa": "ecdsa",
            "paramiko": "paramiko",
            "scp": "scp",
            "sshpubkeys": "sshpubkeys",
            "python-dotenv": "python-dotenv",
            "python-decouple": "python-decouple",
            "environs": "environs",
            "django-environ": "django-environ",
            "python-dotenv[cli]": "python-dotenv[cli]",
            "python-dotenv[parser]": "python-dotenv[parser]",
            "python-dotenv[versioning]": "python-dotenv[versioning]",
            "python-dotenv[validation]": "python-dotenv[validation]",
        }
        self.standard_libs = {"sys", "os", "json", "threading"}  # 排除的标准库
        # 启动输出处理线程
        self.start_output_processor()

    def find_python_interpreter(self):
        """查找系统中的Python解释器"""
        # 检查当前使用的Python解释器
        if os.path.exists(sys.executable):
            return sys.executable

        # 检查常见的Python安装路径
        common_paths = [
            "python.exe", "python3.exe",
            os.path.join(os.environ.get("ProgramFiles", ""), "Python310", "python.exe"),
            os.path.join(os.environ.get("ProgramFiles", ""), "Python39", "python.exe"),
            os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Programs", "Python", "Python310",
                         "python.exe")
        ]

        for path in common_paths:
            if os.path.exists(path) and os.path.isfile(path):
                return path

        return None

    def set_python_path(self, python_path):
        """设置Python解释器路径"""
        if python_path and os.path.exists(python_path):
            self.python_path = python_path
            return True
        return False

    def check_dependencies(self, required_modules):
        """
        新方案：用单行代码检查模块，彻底规避缩进问题。
        功能：检查缺失模块 + 别名映射 + 3次重试 + 环境一致性。
        """
        missing = []
        time.sleep(2)  # 等待系统缓存更新

        # Windows隐藏子进程黑框
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        # 输出检查开始信息
        self.output_queue.put(f"\n=== 开始检查 {len(required_modules)} 个模块 ===")
        print(f"\n=== check_dependencies: 开始检查模块 ===")
        print(f"使用的Python路径: {self.python_path}")

        for module in required_modules:
            # 步骤1：处理“安装包名 → 导入名”的别名映射（保留原有逻辑）
            import_name = module
            for alias, pkg_name in self.module_aliases.items():
                if pkg_name == module:
                    import_name = alias
                    alias_msg = f"注意：模块 '{module}' 实际导入名为 '{import_name}'"
                    self.output_queue.put(alias_msg)
                    print(alias_msg)
                    break

            # 步骤2：3次重试检查（保留原有逻辑，确保结果准确）
            is_installed = False
            for retry in range(3):
                retry_msg = f"检查 '{import_name}'（第 {retry + 1}/3 次）..."
                self.output_queue.put(retry_msg)
                print(retry_msg)

                # 核心修改：用“单行代码”检查模块，无任何缩进问题！
                # 命令逻辑：导入模块 → 成功则打印 SUCCESS（stdout），失败则抛 ImportError（stderr）
                check_cmd = f"import {import_name}; print('SUCCESS')"
                cmd = [self.python_path, "-c", check_cmd]  # 执行单行命令

                try:
                    # 执行检查命令（捕获 stdout 和 stderr）
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        startupinfo=startupinfo,
                        timeout=10  # 超时保护，防止卡住
                    )
                    stdout = result.stdout.strip()
                    stderr = result.stderr.strip()
                    print(f"检查结果：stdout={stdout}, stderr={stderr}")

                    # 步骤3：判断模块是否安装（逻辑简化，更直观）
                    if stdout == "SUCCESS" and not stderr:
                        #  stdout 有 SUCCESS → 模块存在
                        is_installed = True
                        break  # 成功则终止重试
                    elif "ImportError: No module named" in stderr or "ModuleNotFoundError" in stderr:
                        # stderr 有“找不到模块”的错误 → 模块不存在
                        self.output_queue.put(f"'{import_name}' 未安装（找不到模块）")
                        time.sleep(1)  # 重试间隔
                    else:
                        # 其他错误（如模块损坏、导入异常）→ 按“未安装”处理，避免执行时崩溃
                        err_msg = f"'{import_name}' 导入异常：{stderr[:100]}"  # 截取前100字符避免过长
                        self.output_queue.put(err_msg)
                        time.sleep(1)

                except Exception as e:
                    # 执行命令时的异常（如Python路径错误、超时）→ 重试
                    retry_fail_msg = f"'{import_name}' 检查失败（重试）：{str(e)}"
                    self.output_queue.put(retry_fail_msg)
                    print(retry_fail_msg)
                    time.sleep(1)

            # 步骤4：记录缺失模块（保留原有逻辑）
            if not is_installed:
                missing_item = (
                    f"{import_name}（对应安装包：{module}）"
                    if import_name != module
                    else module
                )
                missing.append(missing_item)
                missing_msg = f"❌ {missing_item} 未安装"
                self.output_queue.put(missing_msg)
                print(missing_msg)
            else:
                installed_msg = f"✅ {import_name} 已安装"
                self.output_queue.put(installed_msg)
                print(installed_msg)

        # 输出最终检查结果
        final_msg = f"\n检查完成：共缺失 {len(missing)} 个模块 → {missing}"
        self.output_queue.put(final_msg)
        print(f"\n=== check_dependencies 完成：缺失模块 {missing} ===")
        return missing

    def is_module_installed(self, module_name):
        """检查单个模块是否已安装（更健壮的检查逻辑）"""
        try:
            # 尝试导入模块
            __import__(module_name)
            return True
            print("检查单个模块是否已安装（更健壮的检查逻辑）")
        except ImportError as e:
            # 处理特殊情况：模块存在但导入时依赖其他模块
            if "cannot import name" in str(e) or "No module named" in str(e):
                return False
            return True  # 模块存在但有其他导入错误
        except Exception as e:
            # 模块存在但导入时有错误（视为已安装）
            return True

    def start_output_processor(self):
        """启动输出处理线程，用于更新UI显示"""

        def process_output():
            while True:
                try:
                    item = self.output_queue.get(block=True, timeout=1)
                    if item is None:  # 退出信号
                        break

                    # 更新UI显示
                    self.update_installation_status(item)
                    self.output_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"输出处理错误: {str(e)}")
                    break

        self.output_processor = threading.Thread(target=process_output, daemon=True)
        self.output_processor.start()

    def update_installation_status(self, message):
        """更新安装状态显示（线程安全）"""
        if self.installation_window and hasattr(self, 'status_text'):
            def update():
                self.status_text.config(state=tk.NORMAL)
                self.status_text.insert(tk.END, message + "\n")
                self.status_text.see(tk.END)
                self.status_text.config(state=tk.DISABLED)

            self.parent.after(0, update)

    def install_missing_dependencies(self, missing_modules, mirror=None):
        """安装缺失的依赖项"""
        print("安装缺失的依赖项")
        if not missing_modules:
            return True

        if not self.python_path or not os.path.exists(self.python_path):
            messagebox.showerror("错误", "未找到有效的Python解释器，无法安装依赖")
            return False

        # 使用指定的镜像源或默认镜像源
        mirror = mirror or self.current_mirror
        mirror_url = self.mirror_urls.get(mirror, self.mirror_urls["阿里云"])

        # 显示安装确认对话框
        module_list = "\n".join(f"- {module}" for module in missing_modules)
        response = messagebox.askyesno(
            "缺少依赖",
            f"检测到缺少以下必要模块：\n{module_list}\n\n是否自动安装这些模块？\n(将使用 {mirror} 镜像源)"
        )

        if not response:
            return False

        # 创建安装进度窗口
        self.create_installation_window(missing_modules)

        # 启动安装线程
        install_thread = threading.Thread(
            target=self._perform_installation,
            args=(missing_modules, mirror_url),
            daemon=True
        )
        install_thread.start()

        # 等待安装完成
        install_thread.join()

        # 检查安装结果
        success = True
        failed_modules = []
        for module in missing_modules:
            # 转换为导入名（如 opencv-python → cv2）
            import_name = module
            for alias, pkg_name in self.module_aliases.items():
                if pkg_name == module:
                    import_name = alias
                    break

            # 多次检查（3次）
            is_ok = False
            for _ in range(3):
                if self.is_module_installed(import_name):
                    is_ok = True
                    break
                time.sleep(1)

            if not is_ok:
                success = False
                failed_modules.append(f"{import_name}（安装包名：{module}）")

        # 关闭安装窗口
        if self.installation_window:
            self.parent.after(0, self.installation_window.destroy)
            self.installation_window = None

        if success:
            messagebox.showinfo("成功", "所有缺失的模块已安装完成")
        else:
            failed_modules = self.check_dependencies(missing_modules)
            if failed_modules:
                failed_list = "\n".join(f"- {module}" for module in failed_modules)
                messagebox.showerror(
                    "部分安装失败",
                    f"以下模块安装失败：\n{failed_list}\n\n请尝试手动安装：\n{self.python_path} -m pip install {' '.join(failed_modules)}"
                )

        return success

    def create_installation_window(self, modules):
        """创建安装进度显示窗口"""
        self.installation_window = tk.Toplevel(self.parent)
        self.installation_window.title("安装依赖模块")
        self.installation_window.geometry("600x400")
        self.installation_window.transient(self.parent)
        self.installation_window.grab_set()  # 模态窗口

        # 防止用户关闭窗口
        def on_close():
            if messagebox.askyesno("确认", "安装正在进行中，确定要取消吗？"):
                self.is_installing = False
                self.installation_window.destroy()

        self.installation_window.protocol("WM_DELETE_WINDOW", on_close)

        # 显示正在安装的模块
        ttk.Label(
            self.installation_window,
            text=f"正在安装 {len(modules)} 个模块，请稍候...",
            font=("SimHei", 10)
        ).pack(pady=10, padx=10, anchor=tk.W)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.installation_window,
            variable=self.progress_var,
            mode="indeterminate",
            length=550
        )
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()

        # 状态文本区域
        ttk.Label(
            self.installation_window,
            text="安装进度：",
            font=("SimHei", 10)
        ).pack(pady=5, padx=10, anchor=tk.W)

        self.status_text = tk.Text(
            self.installation_window,
            wrap=tk.WORD,
            font=("SimHei", 9),
            height=12,
            width=70
        )
        self.status_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        self.status_text.config(state=tk.DISABLED)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.status_text, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)

    def _perform_installation(self, modules, mirror_url):
        """执行实际的安装操作"""
        self.is_installing = True
        self.output_queue.put(f"开始安装 {len(modules)} 个模块...")
        self.output_queue.put(f"使用镜像源: {mirror_url}")

        # 先尝试升级pip
        try:
            self.output_queue.put("正在升级pip...")
            upgrade_cmd = [
                self.python_path, "-m", "pip", "install",
                "--upgrade", "pip",
                "-i", mirror_url
            ]

            result = self._run_install_command(upgrade_cmd)
            if result != 0:
                self.output_queue.put("警告：pip升级失败，继续安装模块")
        except Exception as e:
            self.output_queue.put(f"pip升级出错: {str(e)}")

        # 安装模块
        success_count = 0
        total_modules = len(modules)

        for i, module in enumerate(modules, 1):
            if not self.is_installing:  # 检查是否已取消
                self.output_queue.put("安装已被用户取消")
                return

            self.output_queue.put(f"\n安装模块 ({i}/{total_modules}): {module}")

            # 构建安装命令
            install_cmd = [
                self.python_path, "-m", "pip", "install",
                module, "-i", mirror_url
            ]

            # 执行安装命令
            try:
                return_code = self._run_install_command(install_cmd)

                if return_code == 0:
                    success_count += 1
                    self.output_queue.put(f"✓ 成功安装 {module}")
                else:
                    self.output_queue.put(f"✗ 安装 {module} 失败，返回代码: {return_code}")

                    # 尝试不指定版本安装
                    if "==" in module:
                        simple_module = module.split("==")[0]
                        self.output_queue.put(f"尝试安装最新版本: {simple_module}")
                        install_cmd = [
                            self.python_path, "-m", "pip", "install",
                            simple_module, "-i", mirror_url
                        ]
                        return_code = self._run_install_command(install_cmd)
                        if return_code == 0:
                            success_count += 1
                            self.output_queue.put(f"✓ 成功安装最新版本 {simple_module}")
                        else:
                            self.output_queue.put(f"✗ 仍然无法安装 {simple_module}")

            except Exception as e:
                self.output_queue.put(f"安装 {module} 时出错: {str(e)}")

        self.output_queue.put(f"\n安装完成：{success_count}/{total_modules} 个模块安装成功")
        self.is_installing = False

        # 关键修改1：延长等待时间（让系统缓存更新，避免立即检查误判）
        self.output_queue.put("等待系统更新缓存...（3秒后检查）")
        time.sleep(3)

        # 关键修改2：用“导入名”检查（而非安装包名），结合别名映射
        failed_modules = []
        for install_name in modules:
            # 先把“安装包名”转回“导入名”（比如 opencv-python → cv2）
            import_name = install_name
            # 反向查找别名映射：找到安装包名对应的导入名
            for alias, pkg_name in self.module_aliases.items():
                if pkg_name == install_name:
                    import_name = alias
                    break

            # 多次检查（3次），确保结果准确
            is_installed = False
            for _ in range(3):
                if self.is_module_installed(import_name):
                    is_installed = True
                    break
                time.sleep(1)  # 每次检查间隔1秒

            if not is_installed:
                failed_modules.append(f"{import_name}（安装包名：{install_name}）")

        if failed_modules:
            self.output_queue.put(f"警告：部分模块检查显示未安装：{failed_modules}")

    def _run_install_command(self, cmd):
        """运行安装命令并捕获输出（强制UTF-8编码）"""
        try:
            # 设置隐藏窗口标志（Windows）
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # 启动子进程（强制UTF-8编码）
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',  # 强制UTF-8编码
                errors='replace',  # 替换无法解码的字符
                bufsize=1,
                universal_newlines=True,
                startupinfo=startupinfo
            )

            # 实时输出（已处理编码）
            for line in process.stdout:
                if not self.is_installing:  # 检查是否已取消
                    process.terminate()
                    return -1
                self.output_queue.put(line.strip())

            process.wait()
            return process.returncode

        except Exception as e:
            self.output_queue.put(f"执行命令时出错: {str(e)}")
            return -1

    def check_code_for_dependencies(self, code):
        """分析代码，提取可能需要的依赖模块"""
        # 简单的正则表达式匹配import语句
        import_patterns = [
            r'^import\s+(\w+)',  # import module
            r'^from\s+(\w+)\s+import',  # from module import ...
            r'^from\s+(\w+)\.\w+\s+import',  # from module.sub import ...
            r'^from\s+(\w+)\s+import\s+\*'  # from module import *
        ]

        import re
        modules = set()

        # 逐行分析代码
        for line in code.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):  # 跳过注释
                continue

            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1)
                    # 排除Python标准库模块
                    if module not in self.get_standard_library_modules():
                        modules.add(module)

        # 关键修改：将导入名转换为实际安装包名
        install_modules = []
        for module in modules:
            # 检查是否有别名映射
            if module in self.module_aliases:
                install_name = self.module_aliases[module]
                # 记录映射关系，方便用户理解
                self.output_queue.put(f"注意：'{module}' 实际需要安装 '{install_name}'")
                install_modules.append(install_name)
            else:
                install_modules.append(module)

        # 去重
        return list(set(install_modules))

    def get_standard_library_modules(self):
        """返回Python标准库模块列表（简化版）"""
        return {
            'sys', 'os', 'json', 'csv', 're', 'math', 'random', 'datetime',
            'time', 'logging', 'unittest', 'subprocess', 'shutil', 'glob',
            'argparse', 'collections', 'itertools', 'functools', 'pathlib',
            'configparser', 'hashlib', 'base64', 'struct', 'socket', 'select',
            'threading', 'multiprocessing', 'queue', 'tkinter', 'turtle',
            'xml', 'html', 'http', 'urllib', 'email', 'sqlite3', 'decimal',
            'statistics', 'array', 'bisect', 'calendar', 'io', 'tempfile'
        }
    def install_modules(self, missing_modules):
        """新增：安装缺失的模块（用你的镜像源和Python路径）""" #13.19新增
        if not missing_modules:
            return True  # 没有缺失模块，直接返回成功

        # 获取当前镜像源的URL（复用你的mirror_urls）
        mirror_url = self.mirror_urls.get(self.current_mirror, "https://mirrors.aliyun.com/pypi/simple/")
        self.output_queue.put(f"开始安装缺失模块：{', '.join(missing_modules)}")
        self.output_queue.put(f"使用镜像源：{self.current_mirror} ({mirror_url})")

        # 构造pip安装命令（用你设置的Python路径）
        cmd = [
            self.python_path, "-m", "pip", "install",
            "-i", mirror_url,  # 用指定镜像源
            "--upgrade-strategy", "only-if-needed"  # 避免不必要的升级
        ] + missing_modules  # 加上要安装的模块列表

        # 隐藏子进程窗口（Windows专用）
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            # 执行安装命令（实时输出日志）
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 把错误也重定向到stdout，方便统一输出
                text=True,
                bufsize=1,  # 行缓冲，实时输出
                universal_newlines=True,
                startupinfo=startupinfo
            )

            # 实时读取安装日志，输出到队列（用户能看到安装进度）
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break  # 进程结束，退出循环
                if line:
                    self.output_queue.put(line.strip())  # 输出每一行日志

            # 检查安装结果（返回码=0 → 成功）
            if process.returncode == 0:
                self.output_queue.put(f"所有模块安装完成：{', '.join(missing_modules)}")
                return True
            else:
                self.output_queue.put(f"模块安装失败！返回码：{process.returncode}")
                return False

        except Exception as e:
            self.log(f"安装模块时出错: {str(e)}")
            self.output_queue.put(f"安装出错：{str(e)}")
            return False
    def auto_manage_dependencies(self, code):
        """自动管理代码依赖：检测并安装缺失模块"""
        # 分析代码获取可能的依赖
        print('正在分析代码依赖')
        self.output_queue.put("正在分析代码依赖...")
        required_modules = self.check_code_for_dependencies(code)

        if not required_modules:
            self.output_queue.put("未检测到需要安装的第三方模块")
            print('未检测到需要安装的第三方模块')
            return True

        # 检查缺失的模块
        missing_modules = self.check_dependencies(required_modules)

        if not missing_modules:
            self.output_queue.put("所有必要的模块都已安装")
            print('所有必要的模块都已安装')
            return True

        # 安装缺失的模块
        print("【即将调用】准备执行 install_missing_dependencies")  # 新增
        self.output_queue.put("【即将调用】准备安装缺失模块...")  # 新增
        return self.install_missing_dependencies(missing_modules)
        #以下为新增代码


class StreamReader(threading.Thread):
    """读取子进程输出流的线程"""
    def __init__(self, stream, queue, stream_type, parent):
        threading.Thread.__init__(self)
        self.stream = stream
        self.queue = queue
        self.stream_type = stream_type  # 'stdout' 或 'stderr'
        self.running = True
        self.parent = parent  # 保存父类引用，用于处理输入请求
        self.exit_patterns = [r'按任意键退出', r'按回车键退出', r'press any key to exit', r'press enter to exit']

    # def run(self):
    #     buffer = b""  # 用字节缓冲区，避免单字节解码错误
    #     input_patterns = [r'[？?]', r'请输入', r'是否', r'确认', r'继续']
    #     possible_encodings = ['utf-8', 'gbk', 'gb2312', 'cp936']  # 优先尝试的编码
    #
    #     while self.running:
    #         try:
    #             # 一次读取多个字节，提高编码识别准确率
    #             byte_data = self.stream.buffer.read(1024)  # 一次读1024字节
    #             if not byte_data:
    #                 time.sleep(0.1)
    #                 continue
    #
    #             # 累积到缓冲区
    #             buffer += byte_data
    #
    #             # 尝试用多种编码解码整个缓冲区
    #             decoded_text = None
    #             for encoding in possible_encodings:
    #                 try:
    #                     decoded_text = buffer.decode(encoding)
    #                     buffer = b""  # 解码成功，清空缓冲区
    #                     break
    #                 except UnicodeDecodeError as e:
    #                     # 如果是部分可解码，保留未解码的字节
    #                     if e.start == 0:
    #                         continue  # 完全不能解码，尝试下一种编码
    #                     # 部分解码成功
    #                     decoded_text = buffer[:e.start].decode(encoding)
    #                     buffer = buffer[e.start:]  # 保留未解码的部分
    #                     break
    #
    #             # 如果所有编码都失败，逐个字节处理
    #             if decoded_text is None:
    #                 decoded_text = ""
    #                 for b in buffer:
    #                     decoded_text += chr(b) if 32 <= b <= 126 else '�'  # 只保留可见字符
    #                 buffer = b""
    #
    #             # 发送解码后的文本到UI
    #             for char in decoded_text:
    #                 self.queue.put((self.stream_type, char))
    #
    #             # 原有逻辑：检查输入请求等
    #             # ...（保留之前的buffer检查逻辑）
    #
    #         except Exception as e:
    #             error_msg = f"读取流错误（已处理）: {str(e)}"
    #             self.queue.put(('error', error_msg + "\n"))
    #             break
    #         except UnicodeDecodeError as e:
    #             # 专门处理编码错误，避免崩溃
    #             if not hasattr(self, 'encoding_error_shown'):
    #                 self.encoding_error_shown = True
    #                 error_msg = "编码解析错误（已处理）: 部分特殊字符已替换为�，不影响功能"
    #                 self.queue.put(('error', error_msg + "\n"))
    #             # 尝试跳过错误字节继续读取
    #             continue
    #         except Exception as e:
    #             error_msg = f"读取流错误（已处理）: {str(e)}"
    #             self.queue.put(('error', error_msg + "\n"))
    #             break
    def run(self):
        """读取流内容并放入队列，检测输入请求和程序退出提示"""
        buffer = ""
        input_patterns = [
            # 中文基础输入提示
            r'请输入', r'请选择', r'请回答', r'请填写', r'请输入',
            r'请输入[^\n]*:', r'请选择[^\n]*:', r'请输入[^\n]*？',
            r'请输入[^\n]*?', r'请选择[^\n]*？', r'请回答[^\n]*？',

            # 中文确认/选择提示
            r'是否', r'确认', r'继续', r'同意', r'接受', r'允许',
            r'是否[^\n]*？', r'确认[^\n]*？', r'继续[^\n]*？',
            r'同意[^\n]*？', r'接受[^\n]*？', r'允许[^\n]*？',
            r'是/否', r'确认/取消', r'同意/拒绝', r'继续/终止',

            # 中文疑问式提示
            r'是吗？', r'对吗？', r'可以吗？', r'行吗？', r'好吗？',
            r'如何[^\n]*？', r'什么[^\n]*？', r'哪个[^\n]*？', r'哪里[^\n]*？',

            # 英文基础输入提示
            r'enter', r'input', r'please enter', r'please input',
            r'enter[^\n]*:', r'input[^\n]*:', r'enter[^\n]*?',
            r'input[^\n]*?', r'provide[^\n]*:', r'enter your[^\n]*',

            # 英文确认/选择提示
            r'confirm', r'select', r'choose', r'continue', r'accept',
            r'confirm[^\n]*?', r'select[^\n]*?', r'choose[^\n]*?',
            r'continue[^\n]*?', r'accept[^\n]*?', r'yes/no', r'y/n',
            r'yes or no', r'confirm/cancel', r'accept/decline',

            # 中英文混合/符号提示
            r'[Yy]/[Nn]', r'[Yy]es/[Nn]o', r'[Yy]/[Nn]?', r'\[Y/N\]', r'\(y/n\)',
            r'<输入>', r'<选择>', r'\[请输入\]', r'\(请选择\)',
            r'>>', r'> ', r'：', r': '
        ]

        while self.running:
            try:
                # 读取单个字符，而不是整行，以便及时检测输入请求
                char = self.stream.read(1)
                if char:
                    buffer += char
                    self.queue.put((self.stream_type, char))

                    # 检查是否是程序退出提示
                    if any(pattern in buffer for pattern in self.exit_patterns):
                        # 自动发送一个换行符来响应"按任意键退出"
                        self.parent.auto_respond_to_exit()
                        buffer = ""

                    # 检查是否可能需要用户输入
                    elif any(pattern in buffer for pattern in input_patterns) or '\n' in buffer:
                        # 检查缓冲区末尾是否可能是输入提示符
                        if buffer.endswith((':', '：', '?', '？', '>')):
                            # 触发输入请求处理
                            self.parent.request_user_input()
                            buffer = ""  # 重置缓冲区
                        elif '\n' in buffer:
                            # 换行后重置缓冲区
                            buffer = ""
                else:
                    # 流已关闭，等待片刻后再次检查
                    time.sleep(0.1)
                    if not self.running:
                        break
            except Exception as e:
                self.queue.put(('error', f"读取流错误: {str(e)}\n"))
                break
    def stop(self):
        """停止读取线程"""
        self.running = False

class PythonExecutor:
    def __init__(self, root):
        print("=== 开始执行 PythonExecutor 的 __init__ 方法 ===")
        self.root = root
        self.root.title("小川助手")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        # 添加依赖管理器
        self.dependency_manager = DependencyManager(root)
        print("DependencyManager 实例创建成功！")
        # 状态变量
        self.is_running = False
        self.current_task_type = None  # 当前任务类型: None, 'execution', 'ai_generation'
        self.current_process = None  # 当前执行的子进程
        self.reader_threads = []     # 流读取线程
        self.execution_folder = os.getcwd()  # 默认执行文件夹
        self.state_lock = threading.Lock()  # 线程安全锁
        self.environment_checked = False  # 环境检查标记
        self.environment_check_in_progress = False  # 环境检查是否正在进行中
        self.execution_timeout = None  # 执行超时计时器
        self.waiting_for_input = False  # 是否正在等待用户输入
        self.process_monitor = None  # 进程监控线程
        self.current_api_thread = None  # 当前API调用线程
        self.env_check_lock = None  # 环境检查锁
        self.log_file = None  # 日志文件
        
        # 初始化默认配置
        self.llm_config = {
            "api_url": "https://api.openai.com/v1/chat/completions",
            "model_name": "gpt-3.5-turbo",
            "api_key": "",
            "timeout": 120,
            "retry_count": 3,
            "retry_delay": 2,
            "auto_execute": False,  # 是否自动执行生成的代码
            "pypi_mirror": "阿里云",  # PyPI镜像源
            "execution_timeout": 300  # 代码执行超时时间(秒)
        }
        
        # 初始化日志系统
        self.init_logging()
        
        # 尝试加载保存的配置
        self.load_config()
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TButton", font=("SimHei", 10))
        self.style.configure("TLabel", font=("SimHei", 10))
        self.style.configure("TNotebook", font=("SimHei", 10))
        self.style.configure("TProgressbar", thickness=10)
        
        # 创建界面元素
        self.create_widgets()
        
        # 记录最近打开的文件
        self.recent_files = []
        
        # 用于进程间通信的队列
        self.output_queue = queue.Queue()
        self.input_queue = queue.Queue()
        
        # 自动配置环境路径
        self.configure_environment()
        
        # 显示初始信息和状态
        self.show_output("欢迎使用小川助手\n", "info")
        self.show_output("本执行器支持交互式代码执行，可处理需要用户输入的程序\n", "info")
        self.update_main_status("就绪")
        
        # 启动输出处理线程
        self.start_output_processor()
        
        # 检查是否是初次运行，若是则执行环境检查
        self.check_first_run()

    def get_standard_library_modules(self):#9.1 13.00新增代码
        """
        定义Python标准库模块列表（这些模块是Python自带的，不需要安装）
        可以根据你的代码用到的模块，随时补充或删减
        """
        # 常见的Python标准库模块，按类别整理，方便你理解和修改
        standard_modules = {
            # 系统操作相关
            'os', 'sys', 'shutil', 'pathlib',
            # 数据处理相关
            'json', 'csv', 'xml', 'configparser',
            # 时间日期相关
            'time', 'datetime', 'calendar',
            # 线程/进程相关
            'threading', 'multiprocessing', 'queue',
            # 正则表达式
            're',
            # 文件操作相关
            'tempfile', 'io',
            # 其他常见自带模块
            'math', 'random', 'collections', 'itertools',
            'traceback', 'logging', 'enum'
        }
        return standard_modules

    def init_logging(self):
        """初始化日志系统，用于调试闪退问题"""
        try:
            log_dir = "executor_logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            log_filename = os.path.join(log_dir, f"executor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            self.log_file = open(log_filename, "w", encoding="utf-8")
            self.log("日志系统初始化成功")
        except Exception as e:
            # 如果日志初始化失败，尽量在控制台输出
            print(f"日志初始化失败: {str(e)}")
            self.log_file = None

    def log(self, message):
        """记录日志"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{timestamp}] {message}\n"

            # 写入日志文件
            if self.log_file:
                self.log_file.write(log_msg)
                self.log_file.flush()

            # 同时输出到控制台（调试用）
            print(log_msg.strip())
        except:
            pass

    def create_widgets(self):
        # 创建主标签页
        main_notebook = ttk.Notebook(self.root)
        main_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. 主功能标签页
        main_tab = ttk.Frame(main_notebook)
        main_notebook.add(main_tab, text="主功能")
        
        # 2. 配置标签页
        config_tab = ttk.Frame(main_notebook)
        main_notebook.add(config_tab, text="配置")
        
        # ==============================================
        # 主功能标签页内容
        # ==============================================
        
        # 顶部控制区域
        top_control_frame = ttk.Frame(main_tab, padding="10")
        top_control_frame.pack(fill=tk.X)
        
        # 选择执行文件夹按钮
        folder_btn = ttk.Button(top_control_frame, text="选择执行文件夹", command=self.choose_execution_folder)
        folder_btn.pack(side=tk.LEFT, padx=5)
        
        # 显示当前执行文件夹
        self.folder_var = tk.StringVar(value=f"当前执行文件夹: {self.execution_folder}")
        folder_label = ttk.Label(top_control_frame, textvariable=self.folder_var, font=("SimHei", 9))
        folder_label.pack(side=tk.LEFT, padx=5)
        
        # Python解释器路径选择
        python_path_frame = ttk.Frame(top_control_frame)
        python_path_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(python_path_frame, text="Python解释器:").pack(side=tk.LEFT, padx=5)
        self.python_path_var = tk.StringVar(value=self.find_python_interpreter())
        ttk.Entry(python_path_frame, textvariable=self.python_path_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(python_path_frame, text="浏览", command=self.browse_python_interpreter).pack(side=tk.LEFT, padx=5)
        
        # 添加查找Python按钮，解决自动获取失败问题
        ttk.Button(python_path_frame, text="查找Python", command=self.search_python_interpreters).pack(side=tk.LEFT, padx=5)
        
        # 分隔线
        separator0 = ttk.Separator(main_tab, orient="horizontal")
        separator0.pack(fill=tk.X, padx=10)
        
        # 中间区域 - 分为左右两栏
        middle_frame = ttk.Frame(main_tab)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左栏 - 文件执行区域
        left_frame = ttk.LabelFrame(middle_frame, text="文件执行", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 文件路径输入框
        file_path_frame = ttk.Frame(left_frame)
        file_path_frame.pack(fill=tk.X, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_path_frame, textvariable=self.file_path_var)
        file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 浏览按钮
        browse_btn = ttk.Button(file_path_frame, text="浏览文件", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # 执行文件按钮
        self.run_file_btn = ttk.Button(left_frame, text="执行选中文件", command=self.run_script)
        self.run_file_btn.pack(fill=tk.X, pady=5)
        
        # 右栏 - AI代码生成区域
        right_frame = ttk.LabelFrame(middle_frame, text="AI代码生成", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 提示词输入区域
        ttk.Label(right_frame, text="请输入代码生成指令:").pack(anchor=tk.W, pady=5)
        
        self.prompt_text = scrolledtext.ScrolledText(
            right_frame, 
            wrap=tk.WORD, 
            font=("SimHei", 10),
            height=4
        )
        self.prompt_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.prompt_text.insert(tk.END, "把当前根目录下所有文件（含子文件夹内的）剪到根目录顶层，按文件类型（文档、图片等）建分类文件夹并归类文件，最后删除根目录空文件夹。")
        
        # AI生成按钮
        ai_btn_frame = ttk.Frame(right_frame)
        ai_btn_frame.pack(fill=tk.X, pady=5)
        
        self.generate_btn = ttk.Button(ai_btn_frame, text="生成并执行代码", command=self.generate_and_execute)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # 保存代码按钮
        self.save_code_btn = ttk.Button(ai_btn_frame, text="保存生成的代码", command=self.save_generated_code)
        self.save_code_btn.pack(side=tk.LEFT, padx=5)
        
        # 生成的代码显示区域
        ttk.Label(right_frame, text="生成的代码:").pack(anchor=tk.W, pady=5)
        
        self.generated_code_text = scrolledtext.ScrolledText(
            right_frame, 
            wrap=tk.WORD, 
            font=("Consolas", 10),
            bg="#2d2d2d",
            fg="#ffffff",
            height=6
        )
        self.generated_code_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 分隔线
        separator1 = ttk.Separator(main_tab, orient="horizontal")
        separator1.pack(fill=tk.X, padx=10, pady=5)
        
        # 输入和输出区域
        io_frame = ttk.Frame(main_tab)
        io_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 输入区域
        input_frame = ttk.LabelFrame(io_frame, text="程序输入", padding="10")
        input_frame.pack(fill=tk.X, padx=0, pady=(0,5))
        
        input_frame_inner = ttk.Frame(input_frame)
        input_frame_inner.pack(fill=tk.X)
        
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame_inner, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.input_entry.config(state=tk.DISABLED)  # 默认禁用
        
        self.send_input_btn = ttk.Button(input_frame_inner, text="发送输入", command=self.send_user_input)
        self.send_input_btn.pack(side=tk.LEFT)
        self.send_input_btn.config(state=tk.DISABLED)  # 默认禁用
        
        # 新增：执行状态显示区域（在程序输入下方）
        status_display_frame = ttk.LabelFrame(io_frame, text="执行状态", padding="10")
        status_display_frame.pack(fill=tk.X, padx=0, pady=(5,5))
        
        status_display_inner = ttk.Frame(status_display_frame)
        status_display_inner.pack(fill=tk.X)
        
        self.main_status_var = tk.StringVar(value="就绪")
        self.main_status_label = ttk.Label(
            status_display_inner, 
            textvariable=self.main_status_var, 
            font=("SimHei", 10, "bold"),
            foreground="#2c3e50"
        )
        self.main_status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 全局取消任务按钮（在状态显示旁边）
        self.global_cancel_btn = ttk.Button(
            status_display_inner, 
            text="取消当前任务", 
            command=self.cancel_current_task,
            style="Accent.TButton"
        )
        self.global_cancel_btn.pack(side=tk.RIGHT, padx=5)
        self.global_cancel_btn.config(state=tk.DISABLED)
        
        # 输出区域
        output_frame = ttk.LabelFrame(io_frame, text="执行输出", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(5,0))

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            font=("SimHei", 10),  # 这里已经是支持中文的字体，无需修改
            bg="#f5f5f5",
            relief=tk.FLAT
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)
        
        # 进度条
        progress_frame = ttk.Frame(main_tab)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, expand=True, padx=5)
        
        # ==============================================
        # 配置标签页内容
        # ==============================================
        config_frame = ttk.LabelFrame(config_tab, text="大模型API设置", padding="10")
        config_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # API URL
        ttk.Label(config_frame, text="API地址:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.api_url_var = tk.StringVar(value=self.llm_config["api_url"])
        ttk.Entry(config_frame, textvariable=self.api_url_var, width=70).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 模型名称
        ttk.Label(config_frame, text="模型名称:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_name_var = tk.StringVar(value=self.llm_config["model_name"])
        ttk.Entry(config_frame, textvariable=self.model_name_var, width=70).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # API Key
        ttk.Label(config_frame, text="API密钥:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.api_key_var = tk.StringVar(value=self.llm_config["api_key"])
        ttk.Entry(config_frame, textvariable=self.api_key_var, width=70, show="*").grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 超时设置
        ttk.Label(config_frame, text="API超时时间(秒):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.timeout_var = tk.StringVar(value=str(self.llm_config["timeout"]))
        ttk.Entry(config_frame, textvariable=self.timeout_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # 代码执行超时设置
        ttk.Label(config_frame, text="代码执行超时(秒):").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.execution_timeout_var = tk.StringVar(value=str(self.llm_config["execution_timeout"]))
        ttk.Entry(config_frame, textvariable=self.execution_timeout_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # 重试次数
        ttk.Label(config_frame, text="重试次数:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.retry_count_var = tk.StringVar(value=str(self.llm_config["retry_count"]))
        ttk.Entry(config_frame, textvariable=self.retry_count_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # 重试延迟
        ttk.Label(config_frame, text="重试延迟(秒):").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.retry_delay_var = tk.StringVar(value=str(self.llm_config["retry_delay"]))
        ttk.Entry(config_frame, textvariable=self.retry_delay_var, width=10).grid(row=6, column=1, sticky=tk.W, pady=5)
        
        # 自动执行设置
        self.auto_execute_var = tk.BooleanVar(value=self.llm_config["auto_execute"])
        auto_execute_check = ttk.Checkbutton(
            config_frame, 
            text="AI生成代码后自动执行（无需确认）", 
            variable=self.auto_execute_var
        )
        auto_execute_check.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # PyPI镜像源选择
        ttk.Label(config_frame, text="PyPI镜像源:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.pypi_mirror_var = tk.StringVar(value=self.llm_config["pypi_mirror"])
        mirror_combobox = ttk.Combobox(
            config_frame, 
            textvariable=self.pypi_mirror_var,
            values=list(PYPI_MIRRORS.keys()),
            state="readonly",
            width=20
        )
        mirror_combobox.grid(row=8, column=1, sticky=tk.W, pady=5)
        
        # 环境检查按钮
        check_env_btn = ttk.Button(config_frame, text="检查并修复环境", command=self.check_and_fix_environment)
        check_env_btn.grid(row=9, column=0, columnspan=2, pady=10)
        
        # 保存配置按钮
        save_config_btn = ttk.Button(config_frame, text="保存配置", command=self.save_config)
        save_config_btn.grid(row=10, column=0, columnspan=2, pady=10)
        
        # 配置标签样式
        self.output_text.tag_config("error", foreground="#e74c3c")
        self.output_text.tag_config("info", foreground="#3498db")
        self.output_text.tag_config("success", foreground="#2ecc71")
        self.output_text.tag_config("path", foreground="#9b59b6")
        self.output_text.tag_config("input", foreground="#e67e22")
        self.output_text.tag_config("code", foreground="#7f8c8d")
        self.output_text.tag_config("warning", foreground="#f39c12")
        self.output_text.tag_config("cmd", foreground="#27ae60")
        self.output_text.tag_config("env", foreground="#8e44ad")
        self.output_text.tag_config("prompt", foreground="#d35400", font=("SimHei", 10, "bold"))
        
        # 配置强调按钮样式
        self.style.configure("Accent.TButton", font=("SimHei", 10, "bold"), foreground="#e74c3c")

    def find_python_interpreter(self):
        """自动查找系统中的Python解释器路径"""
        self.log("开始查找Python解释器")
        
        # 对于打包后的程序，不使用自身作为Python解释器
        if is_standalone_exe():
            self.log("检测到打包环境，寻找系统Python解释器")
            # 尝试常见的Python路径
            common_paths = [
                "python.exe",
                "python3.exe",
                os.path.join(os.environ.get("PYTHON_HOME", ""), "python.exe"),
                os.path.join(os.environ.get("ProgramFiles", ""), "Python310", "python.exe"),
                os.path.join(os.environ.get("ProgramFiles", ""), "Python39", "python.exe"),
                os.path.join(os.environ.get("ProgramFiles", ""), "Python38", "python.exe"),
                os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Python310", "python.exe"),
                os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Python39", "python.exe"),
                os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Python38", "python.exe"),
                os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Programs", "Python", "Python310", "python.exe"),
                os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Programs", "Python", "Python39", "python.exe"),
                os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Programs", "Python", "Python38", "python.exe")
            ]
            
            # 从环境变量PATH中查找
            path_dirs = os.environ.get("PATH", "").split(os.pathsep)
            for dir_path in path_dirs:
                for exe_name in ["python.exe", "python3.exe"]:
                    python_path = os.path.join(dir_path, exe_name)
                    if python_path not in common_paths:
                        common_paths.append(python_path)
            
            for path in common_paths:
                self.log(f"检查Python路径: {path}")
                if os.path.exists(path) and os.path.isfile(path):
                    self.log(f"找到有效的Python解释器: {path}")
                    return path
                    
            self.log("未找到系统Python解释器")
            return ""
        
        # 未打包的程序，使用当前Python解释器
        current_python = sys.executable
        if os.path.exists(current_python):
            self.log(f"使用当前Python解释器: {current_python}")
            return current_python
            
        self.log("未找到Python解释器")
        return ""

    def search_python_interpreters(self):
        """主动搜索系统中的Python解释器并提供选择"""
        self.log("开始主动搜索Python解释器")
        self.update_main_status("正在搜索Python解释器...", "#3498db")
        
        # 创建搜索线程
        def search_thread():
            python_paths = []
            
            # 搜索常见位置
            search_locations = [
                os.environ.get("ProgramFiles", ""),
                os.environ.get("ProgramFiles(x86)", ""),
                os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Programs", "Python"),
                os.environ.get("SystemRoot", ""),
                os.path.join(os.environ.get("SystemRoot", ""), "System32")
            ]
            
            # 从环境变量添加更多位置
            for dir_path in os.environ.get("PATH", "").split(os.pathsep):
                if dir_path not in search_locations:
                    search_locations.append(dir_path)
            
            # 搜索Python解释器
            for root in search_locations:
                if not root or not os.path.exists(root):
                    continue
                    
                for dirpath, _, filenames in os.walk(root):
                    # 限制搜索深度，避免过慢
                    if dirpath.count(os.sep) - root.count(os.sep) > 3:
                        continue
                        
                    for filename in filenames:
                        if filename.lower() in ["python.exe", "python3.exe"]:
                            full_path = os.path.join(dirpath, filename)
                            if full_path not in python_paths:
                                python_paths.append(full_path)
            
            # 过滤重复项和无效路径
            valid_paths = []
            for path in python_paths:
                if os.path.exists(path) and os.path.isfile(path) and path not in valid_paths:
                    valid_paths.append(path)
            
            # 在UI线程中显示结果
            def show_results():
                if valid_paths:
                    # 创建选择对话框
                    dialog = tk.Toplevel(self.root)
                    dialog.title("选择Python解释器")
                    dialog.geometry("600x400")
                    dialog.transient(self.root)
                    dialog.grab_set()
                    
                    ttk.Label(dialog, text="找到以下Python解释器，请选择一个:").pack(pady=10, padx=10, anchor=tk.W)
                    
                    listbox = tk.Listbox(dialog, width=80, height=15)
                    listbox.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
                    
                    scrollbar = ttk.Scrollbar(listbox, orient="vertical", command=listbox.yview)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    listbox.config(yscrollcommand=scrollbar.set)
                    
                    for path in valid_paths:
                        listbox.insert(tk.END, path)
                    
                    # 选择按钮
                    def select_path():
                        if listbox.curselection():
                            selected = listbox.get(listbox.curselection())
                            self.python_path_var.set(selected)
                            dialog.destroy()
                    
                    btn_frame = ttk.Frame(dialog)
                    btn_frame.pack(pady=10, fill=tk.X)
                    
                    ttk.Button(btn_frame, text="选择", command=select_path).pack(side=tk.RIGHT, padx=10)
                    ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.RIGHT)
                    
                    # 默认选择第一个
                    if valid_paths:
                        listbox.selection_set(0)
                else:
                    messagebox.showinfo("未找到", "未在系统中找到Python解释器，请手动安装Python或指定路径。")
                    self.browse_python_interpreter()  # 直接打开浏览对话框
                
                self.update_main_status("就绪")
            
            self.root.after(0, show_results)
        
        # 启动搜索线程
        threading.Thread(target=search_thread, daemon=True).start()

    def browse_python_interpreter(self):
        """浏览选择Python解释器"""
        python_path = filedialog.askopenfilename(
            title="选择Python解释器",
            filetypes=[("Python解释器", "python.exe;python3.exe"), ("所有文件", "*.*")]
        )
        if python_path:
            self.python_path_var.set(python_path)
            self.log(f"手动选择Python解释器: {python_path}")

    def configure_environment(self):
        """配置Python环境"""
        self.log("开始配置环境")
        
        # 确保当前目录在系统路径中
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.append(current_dir)

        # 处理Tcl数据目录问题
        if is_standalone_exe():
            try:
                # 设置Tcl和Tk库路径
                tcl_dir = os.path.join(sys._MEIPASS, 'tcl')
                tk_dir = os.path.join(sys._MEIPASS, 'tk')
                
                self.log(f"Tcl库路径: {tcl_dir}")
                self.log(f"Tk库路径: {tk_dir}")
                
                # 检查目录是否存在
                tcl_exists = os.path.exists(tcl_dir)
                tk_exists = os.path.exists(tk_dir)
                
                self.log(f"Tcl目录存在: {tcl_exists}")
                self.log(f"Tk目录存在: {tk_exists}")
                
                if tcl_exists and tk_exists:
                    os.environ['TCL_LIBRARY'] = tcl_dir
                    os.environ['TK_LIBRARY'] = tk_dir
                    self.log("已设置TCL_LIBRARY和TK_LIBRARY环境变量")
                else:
                    self.log("Tcl或Tk目录不存在，尝试备选方案")
                    
                    # 备选方案：查找系统中的Tcl/Tk库
                    system_tcl_paths = [
                        os.path.join(os.environ.get("ProgramFiles", ""), "Tcl", "lib", "tcl8.6"),
                        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Tcl", "lib", "tcl8.6"),
                    ]
                    
                    for path in system_tcl_paths:
                        if os.path.exists(path):
                            os.environ['TCL_LIBRARY'] = path
                            self.log(f"使用系统Tcl库: {path}")
                            break
                    
                    system_tk_paths = [
                        os.path.join(os.environ.get("ProgramFiles", ""), "Tcl", "lib", "tk8.6"),
                        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Tcl", "lib", "tk8.6"),
                    ]
                    
                    for path in system_tk_paths:
                        if os.path.exists(path):
                            os.environ['TK_LIBRARY'] = path
                            self.log(f"使用系统Tk库: {path}")
                            break
            except Exception as e:
                error_msg = f"配置Tcl/Tk环境时出错: {str(e)}"
                self.log(error_msg)
                self.show_output(error_msg + "\n", "warning")

    def check_first_run(self):
        """检查是否是初次运行，如果是则执行环境检查"""
        self.log("检查是否为初次运行")
        
        # 检查全局锁，避免重复进行环境检查
        if ENVIRONMENT_CHECK_IN_PROGRESS:
            self.show_output("检测到已有环境检查在进行中，跳过重复检查\n", "info")
            return
            
        try:
            # 检查是否有运行记录
            if not os.path.exists("run_history.json"):
                self.show_output("检测到初次运行，开始环境检查...\n", "env")
                self.update_main_status("正在进行环境检查...")
                # 先检查Python解释器是否已设置
                python_path = self.python_path_var.get()
                if not python_path or not os.path.exists(python_path):
                    self.log("未找到有效的Python解释器，先引导用户设置")
                    # 先显示Python解释器设置对话框
                    self.show_python_setup_dialog()
                else:
                    # 创建环境检查对话框
                    self.create_environment_check_dialog()
            else:
                # 读取运行记录
                with open("run_history.json", "r", encoding="utf-8") as f:
                    history = json.load(f)
                self.show_output(f"上次运行时间: {history.get('last_run', '未知')}\n", "info")
                self.environment_checked = True
        except Exception as e:
            error_msg = f"检查运行记录时出错: {str(e)}"
            self.log(error_msg)
            self.show_output(error_msg + "\n", "error")
            self.update_main_status("就绪")
            # 出错时仍进行环境检查
            self.create_environment_check_dialog()

    def show_python_setup_dialog(self):
        """显示Python解释器设置对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("配置Python解释器")
        dialog.geometry("600x300")
        dialog.transient(self.root)
        dialog.grab_set()  # 模态窗口
        dialog.resizable(False, False)
        
        ttk.Label(
            dialog, 
            text="未检测到Python解释器，这是运行程序所必需的。\n请选择您系统中的Python解释器（python.exe）位置。",
            font=("SimHei", 10),
            wraplength=550
        ).pack(pady=20)
        
        path_frame = ttk.Frame(dialog)
        path_frame.pack(fill=tk.X, padx=20, pady=10)
        
        python_path_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=python_path_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        
        def browse():
            path = filedialog.askopenfilename(
                title="选择Python解释器",
                filetypes=[("Python解释器", "python.exe;python3.exe"), ("所有文件", "*.*")]
            )
            if path:
                python_path_var.set(path)
        
        ttk.Button(path_frame, text="浏览", command=browse).pack(side=tk.LEFT)
        
        # 搜索按钮
        ttk.Button(dialog, text="自动搜索Python解释器", command=lambda: [
            dialog.destroy(),
            self.search_python_interpreters()
        ]).pack(pady=10)
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        
        def confirm():
            path = python_path_var.get()
            if path and os.path.exists(path):
                self.python_path_var.set(path)
                dialog.destroy()
                self.create_environment_check_dialog()
            else:
                messagebox.showerror("错误", "请选择有效的Python解释器")
        
        ttk.Button(btn_frame, text="确认", command=confirm).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def create_environment_check_dialog(self):
        """创建环境检查对话框，增强错误处理防止闪退"""
        # 检查环境检查是否正在进行，如果是则返回
        if self.environment_check_in_progress or ENVIRONMENT_CHECK_IN_PROGRESS:
            return
            
        self.log("创建环境检查对话框")
        
        # 获取环境检查锁
        self.env_check_lock = create_environment_check_lock()
        if not self.env_check_lock:
            self.show_output("已有环境检查在进行中，无法重复启动\n", "warning")
            self.update_main_status("就绪")
            return
            
        self.environment_check_in_progress = True
        
        try:
            # 创建顶层窗口
            env_window = tk.Toplevel(self.root)
            env_window.title("环境检查")
            env_window.geometry("600x400")
            env_window.transient(self.root)
            env_window.grab_set()  # 模态窗口
            env_window.resizable(False, False)
            
            # 窗口关闭时重置标志
            def on_close():
                self.environment_check_in_progress = False
                release_environment_check_lock(self.env_check_lock)
                env_window.destroy()
                
            env_window.protocol("WM_DELETE_WINDOW", on_close)
            
            # 添加说明文本
            ttk.Label(
                env_window, 
                text="初次运行，正在检查必要的运行环境...\n这可能需要几分钟时间，请耐心等待。",
                font=("SimHei", 11),
                wraplength=550
            ).pack(pady=20)
            
            # 进度条
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(env_window, variable=progress_var, mode="indeterminate", length=500)
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            # 输出区域
            output_text = scrolledtext.ScrolledText(
                env_window, 
                wrap=tk.WORD, 
                font=("SimHei", 10),
                height=10,
                width=70
            )
            output_text.pack(pady=10)
            output_text.config(state=tk.DISABLED)
            
            # 日志函数
            def log(message, color="black"):
                self.log(f"环境检查: {message}")
                output_text.config(state=tk.NORMAL)
                output_text.insert(tk.END, message + "\n", color)
                output_text.tag_config("green", foreground="#27ae60")
                output_text.tag_config("red", foreground="#e74c3c")
                output_text.tag_config("blue", foreground="#3498db")
                output_text.see(tk.END)
                output_text.config(state=tk.DISABLED)
                env_window.update_idletasks()
            
            # 在线程中执行环境检查
            def perform_check():
                try:
                    log("开始环境检查...", "blue")
                    
                    # 检查Python解释器
                    python_path = self.python_path_var.get()
                    if not python_path or not os.path.exists(python_path):
                        log(f"未找到Python解释器，尝试自动查找...", "blue")
                        python_path = self.find_python_interpreter()
                        if python_path and os.path.exists(python_path):
                            log(f"找到Python解释器: {python_path}", "green")
                            self.root.after(0, lambda: self.python_path_var.set(python_path))
                        else:
                            log("未找到Python解释器，请手动安装Python后再运行本程序。", "red")
                            log("Python官方下载地址: https://www.python.org/downloads/", "blue")
                            self.root.after(0, lambda: self.update_main_status("就绪"))
                            self.root.after(0, on_close)
                            return
                
                    # 检查必要的包
                    missing_packages = self.check_required_packages(python_path, log)
                    
                    # 安装缺失的包
                    if missing_packages:
                        log(f"发现{len(missing_packages)}个缺失的依赖包，开始安装...", "blue")
                        mirror_name = self.llm_config["pypi_mirror"]
                        mirror_url = PYPI_MIRRORS[mirror_name]
                        log(f"使用{mirror_name}镜像源: {mirror_url}", "blue")
                        
                        success = self.install_packages(python_path, missing_packages, mirror_url, log)
                        if not success:
                            log("安装依赖包失败，尝试使用其他镜像源...", "red")
                            # 尝试其他镜像源
                            for name, url in PYPI_MIRRORS.items():
                                if name != mirror_name:
                                    log(f"尝试使用{name}镜像源: {url}", "blue")
                                    if self.install_packages(python_path, missing_packages, url, log):
                                        success = True
                                        break
                        
                        if not success:
                            log("所有镜像源都尝试过，但安装仍然失败，请手动安装这些包：", "red")
                            for pkg in missing_packages:
                                log(f"pip install {pkg}", "blue")
                    
                    log("环境检查完成！", "green")
                    self.environment_checked = True
                    
                    # 记录运行信息
                    with open("run_history.json", "w", encoding="utf-8") as f:
                        json.dump({
                            "first_run": False,
                            "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }, f, ensure_ascii=False, indent=2)
                    
                    # 延迟关闭窗口
                    time.sleep(2)
                    self.root.after(0, lambda: self.update_main_status("就绪"))
                    self.root.after(0, on_close)
                    
                except Exception as e:
                    error_msg = f"环境检查出错: {str(e)}"
                    log(error_msg, "red")
                    log(traceback.format_exc(), "red")  # 记录详细错误堆栈
                    self.root.after(0, lambda: self.update_main_status("就绪"))
                    self.root.after(0, on_close)
                finally:
                    self.environment_check_in_progress = False
                    release_environment_check_lock(self.env_check_lock)
            
            # 启动检查线程
            check_thread = threading.Thread(target=perform_check, daemon=True)
            check_thread.start()
            
        except Exception as e:
            error_msg = f"创建环境检查对话框时出错: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())
            self.show_output(error_msg + "\n", "error")
            self.environment_check_in_progress = False
            release_environment_check_lock(self.env_check_lock)
            self.update_main_status("就绪")

    def check_required_packages(self, python_path, log_func=None):
        """检查必要的Python包是否已安装"""
        missing = []
        
        # 设置隐藏子进程窗口的标志
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        for pkg_import, pkg_install in REQUIRED_PACKAGES.items():
            # 检查包是否安装
            cmd = [python_path, "-c", f"import {pkg_import}"]
            self.log(f"检查包: {pkg_import}，命令: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    startupinfo=startupinfo,
                    timeout=10  # 设置超时
                )
                
                if result.returncode != 0:
                    missing.append(pkg_install)
                    if log_func:
                        log_func(f"未安装: {pkg_install}", "red")
                else:
                    if log_func:
                        log_func(f"已安装: {pkg_install}", "green")
            except Exception as e:
                self.log(f"检查包 {pkg_import} 时出错: {str(e)}")
                missing.append(pkg_install)
                if log_func:
                    log_func(f"检查{pkg_install}时出错，将尝试安装", "warning")
                    
        return missing

    def install_packages(self, python_path, packages, mirror_url, log_func=None):
        """安装指定的Python包，使用指定的镜像源"""
        try:
            # 对于打包的程序，确保使用正确的Python解释器而不是自身
            if is_standalone_exe() and os.path.basename(python_path).lower() == os.path.basename(sys.executable).lower():
                if log_func:
                    log_func("检测到打包环境，寻找系统Python解释器...", "blue")
                python_path = self.find_python_interpreter()
                if not python_path or not os.path.exists(python_path):
                    if log_func:
                        log_func("找不到系统Python解释器，无法自动安装依赖包", "red")
                    return False
            
            self.log(f"安装包: {packages}，使用Python: {python_path}")
            
            # 设置隐藏子进程窗口的标志
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # 构建pip命令
            cmd = [
                python_path, "-m", "pip", "install",
                "--upgrade", "pip",
                "-i", mirror_url
            ]
            
            # 先升级pip
            if log_func:
                log_func("正在升级pip...", "blue")
                
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    startupinfo=startupinfo,
                    timeout=120  # 2分钟超时
                )
                
                if result.returncode != 0 and log_func:
                    log_func(f"升级pip时出错: {result.stderr}", "red")
                    # 继续安装包，即使pip升级失败
            except Exception as e:
                self.log(f"升级pip时出错: {str(e)}")
                if log_func:
                    log_func(f"升级pip时出错: {str(e)}", "red")
            
            # 安装包
            install_cmd = [
                python_path, "-m", "pip", "install",
                "-i", mirror_url
            ] + packages
            
            if log_func:
                log_func(f"安装命令: {' '.join(install_cmd)}", "blue")
                
            try:
                process = subprocess.Popen(
                    install_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    startupinfo=startupinfo
                )
                
                # 实时输出安装过程
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line and log_func:
                        log_func(line.strip())
                
                return process.returncode == 0
            except Exception as e:
                error_msg = f"安装包时出错: {str(e)}"
                self.log(error_msg)
                if log_func:
                    log_func(error_msg, "red")
                return False
                
        except Exception as e:
            error_msg = f"安装包时出错: {str(e)}"
            self.log(error_msg)
            if log_func:
                log_func(error_msg, "red")
            return False

    def check_and_fix_environment(self):
        """手动触发环境检查和修复"""
        # 检查是否已有环境检查在进行
        if self.is_running or self.environment_check_in_progress or ENVIRONMENT_CHECK_IN_PROGRESS:
            messagebox.showwarning("警告", "当前有任务正在执行或环境检查已在进行中，请等待完成后再试")
            return
            
        # 先检查Python解释器是否已设置
        python_path = self.python_path_var.get()
        if not python_path or not os.path.exists(python_path):
            self.show_python_setup_dialog()
        else:
            self.update_main_status("正在进行环境检查...")
            # 创建环境检查对话框
            self.create_environment_check_dialog()

    def browse_file(self):
        """浏览并选择Python文件"""
        file_path = filedialog.askopenfilename(
            title="选择Python文件",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            # 记录最近打开的文件
            if file_path not in self.recent_files:
                self.recent_files.append(file_path)
                if len(self.recent_files) > 5:  # 只保留最近5个文件
                    self.recent_files.pop(0)
        print(f"浏览文件成功，已保存路径: {file_path}")
        print(f"self.file_path_var 当前值: {self.file_path_var.get()}")
    def choose_execution_folder(self):
        """选择执行文件夹"""
        folder = filedialog.askdirectory(title="选择执行文件夹")
        if folder:
            self.execution_folder = folder
            self.folder_var.set(f"当前执行文件夹: {self.execution_folder}")
            self.show_output(f"已设置执行文件夹为: {self.execution_folder}\n", "info")

    def start_output_processor(self):
        """启动输出处理线程，用于实时显示子进程输出"""
        def process_output():
            while True:
                try:
                    # 阻塞等待队列消息，但设置超时以允许线程退出
                    item = self.output_queue.get(block=True, timeout=1)
                    if item is None:  # 退出信号
                        break
                        
                    stream_type, data = item
                    # 根据流类型显示不同颜色
                    if stream_type == 'stdout':
                        self.show_output(data, "output")
                    elif stream_type == 'stderr':
                        self.show_output(data, "error")
                    elif stream_type == 'cmd':
                        self.show_output(data, "cmd")
                    elif stream_type == 'env':
                        self.show_output(data, "env")
                    elif stream_type == 'input_prompt':
                        self.show_output(data, "prompt")
                    else:
                        self.show_output(data, "error")
                        
                    self.output_queue.task_done()
                except queue.Empty:
                    # 超时，继续循环检查
                    continue
                except Exception as e:
                    self.show_output(f"输出处理错误: {str(e)}\n", "error")
                    break
        
        self.output_processor = threading.Thread(target=process_output, daemon=True)
        self.output_processor.start()

    def update_main_status(self, message, color="#2c3e50"):
        """更新主状态显示（线程安全）"""
        def _update():
            self.main_status_var.set(message)
            self.main_status_label.config(foreground=color)
            self.root.update_idletasks()
        self.root.after(0, _update)

    def set_running_state(self, is_running, task_type=None):
        """设置运行状态（线程安全）"""
        with self.state_lock:
            self.is_running = is_running
            self.current_task_type = task_type
        
        # 确保UI更新在主线程执行
        def update_ui():
            if is_running:
                self.progress_bar.start()
                self.global_cancel_btn.config(state=tk.NORMAL)
                self.run_file_btn.config(state=tk.DISABLED)
                self.generate_btn.config(state=tk.DISABLED)
            else:
                self.progress_bar.stop()
                self.global_cancel_btn.config(state=tk.DISABLED)
                self.run_file_btn.config(state=tk.NORMAL)
                self.generate_btn.config(state=tk.NORMAL)
                self.input_entry.config(state=tk.DISABLED)
                self.send_input_btn.config(state=tk.DISABLED)
                
                # 取消超时计时器
                if self.execution_timeout:
                    self.execution_timeout.cancel()
                    self.execution_timeout = None
                    
                self.waiting_for_input = False
                
        self.root.after(0, update_ui)
        
        # 设置执行超时
        if is_running and task_type == 'execution':
            timeout_seconds = int(self.llm_config.get("execution_timeout", 300))
            self.execution_timeout = threading.Timer(timeout_seconds, self.handle_execution_timeout)
            self.execution_timeout.start()

    def handle_execution_timeout(self):
        """处理代码执行超时"""
        self.root.after(0, lambda: messagebox.showwarning(
            "执行超时", 
            f"代码执行已超过{self.llm_config['execution_timeout']}秒，将被终止。"
        ))
        self.cancel_current_task()

    def cancel_current_task(self):
        """全局取消当前任务，包括代码执行和AI生成"""
        with self.state_lock:
            if not self.is_running:
                return
        
        self.show_output("正在取消当前任务...\n", "info")
        self.update_main_status("正在取消任务...", "#f39c12")
        
        try:
            # 根据任务类型执行不同的取消操作
            if self.current_task_type == 'execution' and self.current_process:
                # 终止代码执行进程
                # 在Windows上终止进程树
                if os.name == 'nt':
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.current_process.pid)])
                else:
                    # 在Unix系统上
                    self.current_process.terminate()
                
                # 等待进程结束
                try:
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.show_output("无法正常终止进程，可能需要手动结束。\n", "warning")
            
            elif self.current_task_type == 'ai_generation' and self.current_api_thread:
                # 终止AI生成线程
                # 由于Python线程无法直接终止，我们设置一个标志并等待线程结束
                if hasattr(self, 'api_cancel_flag'):
                    self.api_cancel_flag = True
                
                # 等待线程结束
                try:
                    if self.current_api_thread.is_alive():
                        self.show_output("正在等待AI生成线程结束...\n", "info")
                        # 最多等待5秒
                        for _ in range(10):
                            if not self.current_api_thread.is_alive():
                                break
                            time.sleep(0.5)
                        
                        if self.current_api_thread.is_alive():
                            self.show_output("AI生成线程无法立即终止，可能需要稍等片刻。\n", "warning")
                except Exception as e:
                    self.show_output(f"取消AI生成时出错: {str(e)}\n", "error")
            
            # 停止所有读取线程
            for thread in self.reader_threads:
                thread.stop()
            self.reader_threads = []
            
            # 重置状态
            self.current_process = None
            self.current_api_thread = None
            self.set_running_state(False)
            self.update_main_status("任务已取消", "#e74c3c")
            self.show_output("\n" + "="*60 + "\n任务已取消\n", "info")
            
        except Exception as e:
            self.show_output(f"取消任务时出错: {str(e)}\n", "error")
            self.set_running_state(False)
            self.update_main_status("取消任务时出错", "#e74c3c")

    def run_script(self, script_code=None, file_path=None):
        """通过CMD执行Python脚本，支持交互式输入"""
        # 检查运行状态
        print("=== run_script 方法被调用了 ===")
        print(f"script_code是否存在: {script_code is not None}")
        print(f"file_path参数: {file_path}")
        # 关键修改1：自动从 self.file_path_var 获取文件路径（如果没传入 file_path）
        if file_path is None or file_path.strip() == "":
            file_path = self.file_path_var.get().strip()  # 从浏览文件保存的路径中读取
            print(f"从 self.file_path_var 自动获取的路径: {file_path}")

        # 关键修改2：检查是否有有效的执行内容（文件路径或代码）
        if not script_code and (not file_path or not os.path.exists(file_path)):
            # 如果既没有代码，也没有有效的文件路径，弹出错误提示
            messagebox.showerror("执行错误", "请先通过「浏览文件」选择有效的 .py 文件！")
            print("错误：没有有效的文件路径或代码，无法执行")
            return

        # 原代码：检查是否有任务正在运行
        with self.state_lock:
            if self.is_running:
                messagebox.showwarning("警告", "已有任务正在执行，请等待完成")
                return
        
        # 验证Python解释器
        python_path = self.python_path_var.get()
        if not python_path or not os.path.exists(python_path):
            messagebox.showerror("错误", "请指定有效的Python解释器路径")
            return
            
        # 创建执行线程，确保UI不会卡死
        execution_thread = threading.Thread(
            target=self._execute_script_thread,
            args=(script_code, file_path, python_path),
            daemon=True
        )
        execution_thread.start()
        print(f"执行线程已启动，将执行文件: {file_path}")
    def auto_respond_to_exit(self):
        """自动响应程序的退出提示"""
        # 将自动响应放入输入队列
        self.input_queue.put("\n")  # 发送一个换行符
        
        # 在UI中显示自动响应
        self.show_output("> 自动响应以结束程序\n", "input")

    def monitor_process(self):
        """监控子进程状态，确保在进程结束后正确清理"""
        while self.is_running and self.current_process is not None:
            # 检查进程是否已结束
            if self.current_process.poll() is not None:
                # 进程已结束，等待一小段时间让输出完成
                time.sleep(0.5)
                
                # 确保所有读取线程已停止
                for thread in self.reader_threads:
                    thread.stop()
                    thread.join(timeout=1)
                
                # 检查返回码
                return_code = self.current_process.returncode
                if return_code == 0:
                    self.show_output(f"\n程序执行完成，返回代码: {return_code}\n", "success")
                else:
                    self.show_output(f"\n程序执行完成，返回代码: {return_code} (可能表示执行出错)\n", "warning")
                
                # 重置状态
                self.current_process = None
                self.reader_threads = []
                self.set_running_state(False)
                self.update_main_status("执行完成", "#2ecc71")
                self.show_output("\n" + "="*60 + "\n执行结束\n", "info")
                break
                
            # 短暂休眠后再次检查
            time.sleep(0.5)

    def _execute_script_thread(self, script_code, file_path, python_path):
        # 显式声明使用全局的os模块
        global os
        
        temp_file = None
        try:
            # 设置运行状态
            self.set_running_state(True, 'execution')
            self.update_main_status("正在执行代码...", "#3498db")
            # ====================== 新增依赖检查代码 ======================
            # 确定需要检查的代码
            print("\n=== 进入 _execute_script_thread 的依赖检查代码块 ===")  # 新增
            code_to_check = script_code
            if not code_to_check and file_path and os.path.exists(file_path):
                # 如果是文件执行，读取文件内容进行检查
                print(f"=== 准备读取文件内容: {file_path} ===")  # 新增
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_to_check = f.read()
                    print(f"=== 读取文件成功，代码长度: {len(code_to_check)} 字符 ===")  # 新增
                    print(f"=== 代码前100字符预览: {code_to_check[:100]} ===")  # 新增（确认读取到内容）
                except Exception as e:
                    print(f"=== 读取文件出错: {str(e)} ===")  # 新增
                    self.show_output(f"读取文件时出错: {str(e)}\n", "error")
                    self.set_running_state(False)
                    self.update_main_status("就绪")
                    return

            # 检查并安装依赖
            if code_to_check:
                print("=== code_to_check 非空，开始调用依赖检查 ===")  # 新增
                self.show_output("正在检查代码依赖...\n", "info")
                # 设置Python路径给依赖管理器
                if self.dependency_manager.set_python_path(python_path):
                    print(f"=== 成功给依赖管理器设置Python路径: {python_path} ===")  # 新增
                else:
                    print(f"=== 给依赖管理器设置Python路径失败: {python_path} ===")  # 新增
                    self.show_output("设置Python路径失败，无法检查依赖\n", "error")
                    self.set_running_state(False)
                    self.update_main_status("就绪")
                    return

                # 调用依赖管理器的核心方法
                print("=== 调用 dependency_manager.auto_manage_dependencies ===")  # 新增
                if not self.dependency_manager.auto_manage_dependencies(code_to_check):
                    self.show_output("依赖安装失败，无法继续执行\n", "error")
                    self.set_running_state(False)
                    self.update_main_status("就绪")
                    return
                else:
                    print("=== 依赖检查和安装完成，准备执行脚本 ===")  # 新增
            else:
                print("=== 跳过依赖检查：code_to_check 为空 ===")  # 新增
            # =============================================================

            if script_code is not None:
                # 如果提供了代码，创建临时文件
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
                temp_file.write(script_code)
                temp_file.close()
                file_path = temp_file.name
                self.show_output(f"已创建临时文件用于执行: {file_path}\n", "info")
            else:
                # 从文件路径获取
                if file_path is None:
                    file_path = self.file_path_var.get()
                    if not file_path or not os.path.exists(file_path):
                        self.show_output("请选择有效的Python文件", "error")
                        self.set_running_state(False)
                        self.update_main_status("就绪")
                        return
        
            # 显示执行信息
            self.show_output("="*60 + "\n", "info")
            self.show_output(f"准备执行: {os.path.basename(file_path)}\n", "info")
            self.show_output(f"文件路径: {file_path}\n", "path")
            self.show_output(f"执行文件夹: {self.execution_folder}\n", "path")
            self.show_output(f"使用的Python解释器: {python_path}\n", "path")
            
            # 构建CMD命令
            cmd = [python_path, file_path]
            self.show_output(f"执行命令: {' '.join(cmd)}\n", "cmd")
            self.show_output("="*60 + "\n", "info")
            
            # 设置隐藏子进程窗口的标志
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            if os.name == 'nt':
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    cwd=self.execution_folder,
                    text=True,
                    bufsize=0,  # 无缓冲，立即输出
                    universal_newlines=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # 允许单独终止
                    startupinfo=startupinfo
                    )
            else:
                # Unix系统使用pty提供更好的交互支持
                import pty
                master, slave = pty.openpty()
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=slave,
                    stdin=slave,
                    stderr=slave,
                    cwd=self.execution_folder,
                    text=True,
                    universal_newlines=True
                    )
                # 将pty的master端作为stdout（指定UTF-8编码）
                self.process_stdout = os.fdopen(master, 'r', encoding='utf-8', errors='replace')
                self.process_stdin = os.fdopen(master, 'w', encoding='utf-8', errors='replace')
            
            # 启动流读取线程
            if os.name == 'nt':
                self.reader_threads = [
                    StreamReader(self.current_process.stdout, self.output_queue, 'stdout', self),
                    StreamReader(self.current_process.stderr, self.output_queue, 'stderr', self)
                ]
            else:
                self.reader_threads = [
                    StreamReader(self.process_stdout, self.output_queue, 'stdout', self)
                ]
            
            for thread in self.reader_threads:
                thread.start()
            
            # 启动进程监控线程
            self.process_monitor = threading.Thread(target=self.monitor_process, daemon=True)
            self.process_monitor.start()
            
            # 处理输入的线程
            self.input_handler_thread = threading.Thread(target=self.process_input_queue, daemon=True)
            self.input_handler_thread.start()
            
            # 等待监控线程完成
            if self.process_monitor:
                self.process_monitor.join()
                
        except Exception as e:
            self.show_output(f"执行过程中出错: {str(e)}\n", "error")
            traceback.print_exc()
        finally:
            # 清理临时文件
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                    self.show_output(f"已删除临时文件: {temp_file.name}\n", "info")
                except Exception as e:
                    self.show_output(f"删除临时文件失败: {str(e)}\n", "warning")
            
            # 关闭文件描述符
            if os.name != 'nt' and hasattr(self, 'process_stdout'):
                try:
                    self.process_stdout.close()
                except:
                    pass
            if os.name != 'nt' and hasattr(self, 'process_stdin'):
                try:
                    self.process_stdin.close()
                except:
                    pass
            
            # 确保状态被正确重置
            if self.is_running:
                self.current_process = None
                for thread in self.reader_threads:
                    thread.stop()
                self.reader_threads = []
                self.set_running_state(False)
                self.update_main_status("就绪")
                self.show_output("\n" + "="*60 + "\n执行结束\n", "info")

    def request_user_input(self):
        """请求用户输入（线程安全）"""
        if self.waiting_for_input or not self.is_running:
            return
            
        self.waiting_for_input = True
        self.update_main_status("等待用户输入...", "#f39c12")
        
        def enable_input():
            self.input_entry.config(state=tk.NORMAL)
            self.send_input_btn.config(state=tk.NORMAL)
            self.input_entry.focus_set()  # 聚焦到输入框
        
        self.root.after(0, enable_input)

    def send_user_input(self):
        """发送用户输入到子进程"""
        user_input = self.input_var.get() + "\n"  # 添加换行符模拟回车
        
        # 在输出区域显示用户输入
        self.show_output(f"> {user_input}", "input")
        
        # 将输入发送到子进程
        self.input_queue.put(user_input)
        
        # 清空输入框并禁用，等待下一次输入请求
        self.input_var.set("")
        self.input_entry.config(state=tk.DISABLED)
        self.send_input_btn.config(state=tk.DISABLED)
        
        self.waiting_for_input = False
        self.update_main_status("正在执行代码...", "#3498db")

    def process_input_queue(self):
        """处理输入队列，将用户输入发送到子进程"""
        while self.is_running and (self.current_process is None or self.current_process.poll() is None):
            try:
                # 检查是否有输入需要发送
                user_input = self.input_queue.get(block=True, timeout=0.5)
                
                if self.current_process and self.current_process.poll() is None:
                    # 发送输入到子进程
                    if os.name == 'nt':
                        self.current_process.stdin.write(user_input)
                        self.current_process.stdin.flush()
                    else:
                        if hasattr(self, 'process_stdin'):
                            self.process_stdin.write(user_input)
                            self.process_stdin.flush()
                
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.output_queue.put(('error', f"发送输入时出错: {str(e)}\n"))
                break

    def show_output(self, text, output_type="output"):
        """在输出区域显示文本（线程安全）"""
        def update_ui():
            self.output_text.config(state=tk.NORMAL)
            
            # 根据输出类型插入文本
            if output_type == "error":
                self.output_text.insert(tk.END, text, "error")
            elif output_type == "info":
                self.output_text.insert(tk.END, text, "info")
            elif output_type == "success":
                self.output_text.insert(tk.END, text, "success")
            elif output_type == "path":
                self.output_text.insert(tk.END, text, "path")
            elif output_type == "input":
                self.output_text.insert(tk.END, text, "input")
            elif output_type == "code":
                self.output_text.insert(tk.END, text, "code")
            elif output_type == "warning":
                self.output_text.insert(tk.END, text, "warning")
            elif output_type == "cmd":
                self.output_text.insert(tk.END, text, "cmd")
            elif output_type == "env":
                self.output_text.insert(tk.END, text, "env")
            elif output_type == "prompt":
                self.output_text.insert(tk.END, text, "prompt")
            else:
                self.output_text.insert(tk.END, text)
                
            self.output_text.config(state=tk.DISABLED)
            self.output_text.see(tk.END)  # 自动滚动到最后
            
        self.root.after(0, update_ui)

    def save_config(self):
        """保存配置，包括新增的自动执行和镜像源设置"""
        try:
            self.llm_config.update({
                "api_url": self.api_url_var.get(),
                "model_name": self.model_name_var.get(),
                "api_key": self.api_key_var.get(),
                "timeout": int(self.timeout_var.get()),
                "retry_count": int(self.retry_count_var.get()),
                "retry_delay": int(self.retry_delay_var.get()),
                "auto_execute": self.auto_execute_var.get(),
                "pypi_mirror": self.pypi_mirror_var.get(),
                "execution_timeout": int(self.execution_timeout_var.get())
            })

            with open("llm_config.json", "w", encoding="utf-8") as f:
                json.dump(self.llm_config, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("成功", "配置已保存")
            self.show_output("配置已更新并保存\n", "success")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {str(e)}")

    def load_config(self):
        """加载保存的配置"""
        try:
            if os.path.exists("llm_config.json"):
                with open("llm_config.json", "r", encoding="utf-8") as f:
                    saved_config = json.load(f)
                    
                for key in saved_config:
                    if key in self.llm_config:
                        self.llm_config[key] = saved_config[key]
        except Exception as e:
            print(f"加载配置失败: {str(e)}")

    def generate_and_execute(self):
        """生成代码并根据设置决定是否自动执行"""
        with self.state_lock:
            if self.is_running:
                messagebox.showwarning("警告", "已有任务正在执行，请等待完成")
                return
            
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("警告", "请输入代码生成指令")
            return
            
        if not self.llm_config["api_url"] or not self.llm_config["model_name"] or not self.llm_config["api_key"]:
            messagebox.showwarning("警告", "请先配置大模型API信息")
            return
            
        # 设置取消标志
        self.api_cancel_flag = False
        
        # 启动API调用线程
        self.current_api_thread = threading.Thread(
            target=self._call_llm_api, 
            args=(prompt,),
            daemon=True
        )
        self.current_api_thread.start()

    def _call_llm_api(self, prompt):
        """调用大模型API生成代码，根据设置决定是否自动执行"""
        try:
            self.set_running_state(True, 'ai_generation')
            self.update_main_status(f"正在向{self.llm_config['model_name']}发送指令...", "#3498db")
            
            # 构建提示词
            full_prompt = f"""请根据以下指令生成Python代码，仅返回可执行的代码部分，不要添加额外解释:
{prompt}

代码应:
1. 直接可执行，不需要修改
2. 包含必要的导入语句
3. 处理可能的异常
4. 在当前目录下执行
5. 有适当的输出以便查看结果
"""
            
            # 检查是否已取消
            if self.api_cancel_flag:
                raise Exception("任务已被用户取消")
            
            # 调用API
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # 创建带有重试机制的会话
            session = requests.Session()
            retry_strategy = Retry(
                total=self.llm_config["retry_count"],
                backoff_factor=self.llm_config["retry_delay"],
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_config['api_key']}"
            }
            
            data = {
                "model": self.llm_config["model_name"],
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 0.7
            }
            
            # 检查是否已取消
            if self.api_cancel_flag:
                raise Exception("任务已被用户取消")
            
            self.update_main_status(f"正在等待{self.llm_config['model_name']}生成代码...", "#3498db")
            
            response = session.post(
                self.llm_config["api_url"],
                headers=headers,
                json=data,
                timeout=self.llm_config["timeout"]
            )
            
            # 检查是否已取消
            if self.api_cancel_flag:
                raise Exception("任务已被用户取消")
            
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
                
            # 解析响应
            result = response.json()
            generated_code = result["choices"][0]["message"]["content"].strip()
            
            # 清理代码
            if generated_code.startswith("```python"):
                generated_code = generated_code[len("```python"):].strip()
            if generated_code.endswith("```"):
                generated_code = generated_code[:-len("```")].strip()
                
            # 在UI中显示生成的代码
            self.root.after(0, lambda: self.generated_code_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.generated_code_text.insert(tk.END, generated_code))
            
            self.show_output(f"成功从{self.llm_config['model_name']}获取代码\n", "success")
            
            # 根据设置决定是否自动执行
            auto_execute = self.llm_config.get("auto_execute", False)
            
            def finalize():
                self.set_running_state(False)
                self.update_main_status("代码生成完成", "#2ecc71")
                
                if auto_execute:
                    # 自动执行代码
                    self.show_output("根据设置，自动执行生成的代码...\n", "info")
                    self.run_script(script_code=generated_code)
                else:
                    # 询问用户是否执行
                    if messagebox.askyesno("执行代码", "是否执行生成的代码?"):
                        self.run_script(script_code=generated_code)
                    else:
                        self.update_main_status("已取消执行")
                        
            self.root.after(0, finalize)
            
        except Exception as e:
            # 忽略取消导致的异常
            if "任务已被用户取消" not in str(e):
                error_msg = f"调用大模型API失败: {str(e)}\n"
                self.show_output(error_msg, "error")
            
            self.set_running_state(False)
            self.update_main_status("代码生成失败" if "取消" not in str(e) else "任务已取消", "#e74c3c")

    def save_generated_code(self):
        """保存生成的代码为Python文件"""
        code = self.generated_code_text.get(1.0, tk.END).strip()
        if not code:
            messagebox.showwarning("警告", "没有可保存的代码，请先生成代码")
            return
            
        # 获取默认文件名
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        default_filename = "generated_code.py"
        if prompt:
            words = prompt.split()[:3]
            if words:
                default_filename = "_".join(words) + ".py"
                default_filename = "".join([c for c in default_filename if c.isalnum() or c in "_-."])
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            title="保存生成的代码",
            initialfile=default_filename
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                messagebox.showinfo("成功", f"代码已保存到 {file_path}")
                self.show_output(f"生成的代码已保存到: {file_path}\n", "success")
                # 更新文件路径输入框，方便用户执行
                self.file_path_var.set(file_path)
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def on_closing(self):
        """窗口关闭时的处理"""
        if self.is_running:
            if messagebox.askyesno("确认关闭", "当前有任务正在执行，确定要关闭吗？"):
                self.cancel_current_task()
                # 等待任务取消完成
                time.sleep(1)
                # 更新运行记录
                try:
                    history = {}
                    if os.path.exists("run_history.json"):
                        with open("run_history.json", "r", encoding="utf-8") as f:
                            history = json.load(f)
                    history["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("run_history.json", "w", encoding="utf-8") as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                except:
                    pass
                # 关闭日志文件
                if self.log_file:
                    self.log("程序关闭")
                    self.log_file.close()
                self.root.destroy()
        else:
            # 更新运行记录
            try:
                history = {}
                if os.path.exists("run_history.json"):
                    with open("run_history.json", "r", encoding="utf-8") as f:
                        history = json.load(f)
                history["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("run_history.json", "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
            except:
                pass
            # 关闭日志文件
            if self.log_file:
                self.log("程序关闭")
                self.log_file.close()
            self.root.destroy()

if __name__ == "__main__":
    # 检查是否已有实例在运行环境检查
    # 新增：打印当前运行的文件路径
    print(f"当前运行的文件: {__file__}")
    print("准备创建主窗口...")
    if os.path.exists(ENVIRONMENT_CHECK_LOCK_FILE):
        try:
            file_mtime = os.path.getmtime(ENVIRONMENT_CHECK_LOCK_FILE)
            # 如果锁文件超过10分钟，视为过期并删除
            if time.time() - file_mtime > 600:
                os.remove(ENVIRONMENT_CHECK_LOCK_FILE)
            else:
                # 有活跃的环境检查，不启动新实例
                sys.exit(0)
        except:
            pass
            
    root = tk.Tk()
    app = PythonExecutor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # 窗口关闭事件
    root.mainloop()
    