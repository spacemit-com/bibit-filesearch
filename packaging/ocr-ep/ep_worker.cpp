// SPDX-License-Identifier: MIT
//
// ep_worker —— 常驻 ONNX Runtime 推理进程，走 SpaceMIT NPU（SpaceMITExecutionProvider）。
//
// ★ 为什么需要这个 C++ 进程：
//   Python 的 `spacemit_ort` shim 在本机建 EP 会话**必然段错误**（实测 5 种写法全 SIGSEGV，
//   子进程退出码 -11），根因是 ABI 不匹配：
//     /usr/lib/libspacemit_ep.so 导出 GetSpaceMITSharedProviderFactory（旧版 shared-provider ABI）
//     onnxruntime 1.24.2 的 register_execution_provider_library() 只认 CreateEpFactories（新版 hosted EP）
//     → register 直接失败：undefined symbol: CreateEpFactories
//   厂商实际支持的入口是 C API `OrtSessionOptionsSpaceMITEnvInit()`（声明见
//   /usr/include/spacemit_ort_env.h），它在 env init 阶段把 EP 装进 ORT 工厂注册表，
//   所以 C++ 侧建会话正常且 NPU 加速生效。
//
// 协议（stdin/stdout 二进制，行头 + 原始字节）：
//   请求 LOAD\t<model_path>\n           → 响应 OK\t<providers>\n  或  ERR\t<msg>\n
//   请求 RUN\t<shape,逗号分隔>\t<dtype>\n + 原始张量字节
//                                    → 响应 OK\t<shape_csv>\t<dtype>\n + 原始张量字节
//                                       或    ERR\t<msg>\n
//   请求 QUIT\n → 退出
//   所有响应后 flush。
//
// 编译（见 debian/rules）：
//   g++ -O2 -std=c++17 ep_worker.cpp -I/usr/include -L/usr/lib \
//       -lonnxruntime -lspacemit_ep -Wl,-rpath,/usr/lib -o ep_worker
//   ★ -lspacemit_ep 必须有：OrtSessionOptionsSpaceMITEnvInit 的符号只在 libspacemit_ep.so 里。

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "spacemit_ort_env.h"

namespace {

std::unique_ptr<Ort::Env> g_env;
std::unique_ptr<Ort::Session> g_session;
std::vector<std::string> g_input_names;
std::vector<std::string> g_output_names;

std::string read_line() {
  std::string line;
  if (!std::getline(std::cin, line)) {
    return "";
  }
  return line;
}

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char ch : s) {
    if (ch == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(ch);
    }
  }
  out.push_back(cur);
  return out;
}

void send(const std::string &s) {
  std::fwrite(s.data(), 1, s.size(), stdout);
  std::fflush(stdout);
}

void send_err(const std::string &msg) {
  // ★ ORT 的异常文本是**多行**的，直接发会把协议行撕碎（客户端 readline 只拿到第一行，
  //   剩下几行被当成下一条响应）。所以把换行/制表符转义掉，保证响应恒为单行。
  std::string s = "ERR\t";
  for (char c : msg) {
    if (c == '\n') {
      s += "\\n";
    } else if (c == '\r') {
      s += "\\r";
    } else if (c == '\t') {
      s += "\\t";
    } else {
      s += c;
    }
  }
  s += "\n";
  send(s);
}

size_t dtype_size(const std::string &dtype) {
  if (dtype == "float32" || dtype == "int32") return 4;
  if (dtype == "float64" || dtype == "int64") return 8;
  if (dtype == "float16" || dtype == "int16") return 2;
  if (dtype == "uint8" || dtype == "int8") return 1;
  return 0;
}

std::string onnx_to_dtype(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
    default: return "unsupported";
  }
}

// 按 dtype 建 tensor；输入只用到 float32（PP-OCR det/cls/rec 都是 float 输入）。
Ort::Value make_tensor(void *data, size_t count, const std::vector<int64_t> &shape,
                       const std::string &dtype) {
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  if (dtype == "float32") {
    return Ort::Value::CreateTensor<float>(mem, static_cast<float *>(data), count,
                                           shape.data(), shape.size());
  }
  if (dtype == "int64") {
    return Ort::Value::CreateTensor<int64_t>(mem, static_cast<int64_t *>(data), count,
                                             shape.data(), shape.size());
  }
  if (dtype == "int32") {
    return Ort::Value::CreateTensor<int32_t>(mem, static_cast<int32_t *>(data), count,
                                             shape.data(), shape.size());
  }
  throw std::runtime_error("make_tensor: unsupported dtype " + dtype);
}

bool handle_load(const std::string &model_path) {
  try {
    Ort::SessionOptions so;
    // ★ 这一句完成 EP 安装 —— 之后**不要**再 AppendExecutionProvider("SpaceMITExecutionProvider")，
    //   否则抛 Unknown provider name。
    Ort::SessionOptionsSpaceMITEnvInit(so);
    g_session = std::make_unique<Ort::Session>(*g_env, model_path.c_str(), so);

    Ort::AllocatorWithDefaultOptions alloc;
    g_input_names.clear();
    g_output_names.clear();
    for (size_t i = 0; i < g_session->GetInputCount(); ++i) {
      auto n = g_session->GetInputNameAllocated(i, alloc);
      g_input_names.emplace_back(n.get());
    }
    for (size_t i = 0; i < g_session->GetOutputCount(); ++i) {
      auto n = g_session->GetOutputNameAllocated(i, alloc);
      g_output_names.emplace_back(n.get());
    }

    auto providers = Ort::GetAvailableProviders();
    std::string pv;
    for (const auto &p : providers) {
      if (!pv.empty()) pv += ",";
      pv += p;
    }
    send("OK\t" + pv + "\n");
    return true;
  } catch (const Ort::Exception &e) {
    send_err(std::string("load failed: ") + e.what());
    return false;
  } catch (const std::exception &e) {
    send_err(std::string("load failed: ") + e.what());
    return false;
  }
}

bool handle_run(const std::string &shape_csv, const std::string &dtype) {
  if (!g_session) {
    send_err("no model loaded");
    return false;
  }
  size_t esz = dtype_size(dtype);
  if (esz == 0) {
    send_err("unsupported input dtype " + dtype);
    return false;
  }

  std::vector<int64_t> shape;
  size_t count = 1;
  for (const auto &tok : split(shape_csv, ',')) {
    if (tok.empty()) continue;
    int64_t d = std::stoll(tok);
    shape.push_back(d);
    count *= static_cast<size_t>(d);
  }
  if (count == 0) {
    send_err("empty shape");
    return false;
  }

  std::vector<uint8_t> buf(count * esz);
  std::cin.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
  if (static_cast<size_t>(std::cin.gcount()) != buf.size()) {
    send_err("short read of input tensor");
    return false;
  }

  try {
    Ort::Value in = make_tensor(buf.data(), count, shape, dtype);
    std::vector<const char *> in_names, out_names;
    for (auto &n : g_input_names) in_names.push_back(n.c_str());
    for (auto &n : g_output_names) out_names.push_back(n.c_str());

    Ort::RunOptions run_opts;
    auto outs = g_session->Run(run_opts, in_names.data(), &in, 1, out_names.data(),
                               out_names.size());
    if (outs.empty()) {
      send_err("no output");
      return false;
    }

    Ort::Value &o = outs[0];
    auto info = o.GetTensorTypeAndShapeInfo();
    auto oshape = info.GetShape();
    auto odtype = onnx_to_dtype(info.GetElementType());
    if (odtype == "unsupported") {
      send_err("unsupported output dtype");
      return false;
    }
    size_t ocount = info.GetElementCount();
    size_t osz = dtype_size(odtype);
    const void *optr = o.GetTensorRawData();

    std::ostringstream hdr;
    hdr << "OK\t";
    for (size_t i = 0; i < oshape.size(); ++i) {
      if (i) hdr << ",";
      hdr << oshape[i];
    }
    hdr << "\t" << odtype << "\n";
    send(hdr.str());
    std::fwrite(optr, 1, ocount * osz, stdout);
    std::fflush(stdout);
    return true;
  } catch (const Ort::Exception &e) {
    send_err(std::string("run failed: ") + e.what());
    return false;
  } catch (const std::exception &e) {
    send_err(std::string("run failed: ") + e.what());
    return false;
  }
}

}  // namespace

int main(int argc, char **argv) {
  // 关闭 stderr 之外的杂音；stdout 是协议通道，绝不能混入日志
  g_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "ep_worker");
  std::cerr << "[worker] SpaceMIT env ready, providers: ";
  for (const auto &p : Ort::GetAvailableProviders()) std::cerr << p << " ";
  std::cerr << std::endl;

  if (argc > 2 && std::string(argv[1]) == "--load") {
    if (!handle_load(argv[2])) return 1;
  }

  std::string line;
  while ((line = read_line()).size() > 0) {
    if (line.rfind("LOAD\t", 0) == 0) {
      handle_load(line.substr(5));
    } else if (line.rfind("RUN\t", 0) == 0) {
      auto parts = split(line, '\t');
      if (parts.size() < 3) {
        send_err("bad RUN header");
      } else {
        handle_run(parts[1], parts[2]);
      }
    } else if (line == "QUIT") {
      break;
    } else {
      send_err("unknown command: " + line);
    }
  }
  return 0;
}
