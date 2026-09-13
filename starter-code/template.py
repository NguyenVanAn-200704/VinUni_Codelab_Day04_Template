"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
import sys
import os

# Đảm bảo in đúng ký tự tiếng Việt trên môi trường Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng cho hệ sinh thái Vingroup (VinFast, Vinpearl).
- Phong cách giao tiếp: Chuyên nghiệp, lịch sự, thân thiện, tận tâm và chính xác tuyệt đối.

## 2. AVAILABLE TOOLS
Bạn có quyền truy cập vào các công cụ sau:
1. `search_product_catalog(category, max_price)`:
   - Tra cứu danh mục sản phẩm, dịch vụ Vingroup theo loại ('xe_dien' cho VinFast, 'du_lich' cho Vinpearl) và mức giá tối đa (VNĐ).
2. `submit_support_ticket(customer_name, issue_description, priority)`:
   - Ghi nhận yêu cầu hỗ trợ, khiếu nại, phản hồi hoặc báo cáo sự cố của khách hàng vào hệ thống chăm sóc khách hàng. Mức độ ưu tiên: 'low', 'medium', 'high'.

## 3. CORE RULES
1. Tuyệt đối KHÔNG BAO GIỜ bịa đặt thông tin sản phẩm, thông số kỹ thuật, giá bán hoặc chính sách không có căn cứ.
2. BẮT BUỘC gọi công cụ `search_product_catalog` khi người dùng cần tra cứu sản phẩm hoặc kiểm tra giá/ngân sách.
3. BẮT BUỘC gọi công cụ `submit_support_ticket` khi người dùng phản ánh sự cố, khiếu nại chất lượng hoặc yêu cầu hỗ trợ kỹ thuật.
4. Chỉ sử dụng thông tin thu được từ kết quả thực thi công cụ (Observation) để tổng hợp câu trả lời cho người dùng.

## 4. OPERATIONAL BOUNDARIES
- Chỉ trả lời các câu hỏi liên quan đến sản phẩm, dịch vụ, ưu đãi và chính sách thuộc hệ sinh thái Vingroup.
- Nếu câu hỏi nằm ngoài phạm vi hoạt động, lịch sự từ chối và hướng dẫn người dùng liên hệ kênh phù hợp.

## 5. OUTPUT CONTRACT
Khi xử lý yêu cầu cần công cụ, tuân thủ nghiêm ngặt quy trình ReAct:
- Thought: Phân tích nhu cầu của người dùng, xác định công cụ và tham số cần thiết.
- Action: Tên công cụ cần gọi (`search_product_catalog` hoặc `submit_support_ticket`).
- Action Input: Các tham số truyền vào công cụ dưới dạng JSON.
- Observation: Dữ liệu thực tế trả về từ công cụ.
- Final Answer: Phản hồi hoàn chỉnh, rõ ràng, chi tiết và thân thiện gửi đến người dùng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, bool]:
        """Phân tích intent độc lập từ user_input (tránh bẫy if-elif)."""
        lower = user_input.lower()
        needs_ticket = False
        needs_catalog = False
        is_faq = False

        # Kiểm tra nhu cầu tạo ticket / báo sự cố
        ticket_indicators = ["lỗi", "hỏng", "sự cố", "khiếu nại", "ẩm mốc", "xử lý gấp", "phản hồi", "báo lỗi", "hỗ trợ"]
        name_indicators = ["tôi tên", "tên tôi là", "khách hàng"]
        if any(k in lower for k in ticket_indicators) and (any(n in lower for n in name_indicators) or "bị lỗi" in lower or "ẩm mốc" in lower):
            needs_ticket = True

        # Kiểm tra câu hỏi chính sách/FAQ
        is_asking_policy = ("chính sách" in lower or "bao lâu" in lower or "mấy năm" in lower or "thế nào" in lower) and ("bảo hành" in lower)

        # Kiểm tra nhu cầu tra cứu sản phẩm
        catalog_triggers = ["xem", "tìm", "mua", "tham khảo", "bảng giá", "giá dưới", "có xe", "có resort", "có phòng", "ngân sách"]
        has_price = bool(re.search(r"\d+\s*(?:triệu|tr|tỷ|ty|vnd|đ)", lower) or re.search(r"dưới\s*\d+", lower))
        
        if not is_asking_policy and (any(k in lower for k in catalog_triggers) or has_price):
            if any(c in lower for c in ["xe", "vinfast", "vf", "resort", "vinpearl", "khách sạn", "du lịch", "nghỉ dưỡng", "phòng"]):
                needs_catalog = True

        if not needs_catalog and not needs_ticket:
            is_faq = True
        elif is_asking_policy and not needs_catalog and not needs_ticket:
            is_faq = True

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }

    def _extract_catalog_args(self, user_input: str) -> Dict[str, Any]:
        """Trích xuất tham số cho công cụ search_product_catalog."""
        lower = user_input.lower()
        if any(w in lower for w in ["resort", "vinpearl", "khách sạn", "du lịch", "du_lich", "nghỉ dưỡng", "phòng", "nha trang", "phú quốc", "landmark 81"]):
            category = "du_lich"
        else:
            category = "xe_dien"

        max_price = 999999999999
        trieu_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:triệu|tr|m\b)", lower)
        ty_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:tỷ|ty|b\b)", lower)
        raw_num_match = re.search(r"dưới\s*(\d{7,})", lower)

        if trieu_match:
            val = float(trieu_match.group(1).replace(",", "."))
            max_price = int(val * 1_000_000)
        elif ty_match:
            val = float(ty_match.group(1).replace(",", "."))
            max_price = int(val * 1_000_000_000)
        elif raw_num_match:
            max_price = int(raw_num_match.group(1))

        return {
            "category": category,
            "max_price": max_price
        }

    def _extract_ticket_args(self, user_input: str) -> Dict[str, Any]:
        """Trích xuất tham số cho công cụ submit_support_ticket."""
        name_match = re.search(
            r"(?:tôi tên là|tôi tên|tên tôi là|khách hàng)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|–|\bxe\b|\bphòng\b|\bvà\b|$)",
            user_input,
            re.IGNORECASE
        )
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

        lower = user_input.lower()
        if any(w in lower for w in ["gấp", "nghiêm trọng", "khẩn cấp", "high", "ngay lập tức"]):
            priority = "high"
        elif any(w in lower for w in ["thấp", "low", "nhẹ", "không gấp"]):
            priority = "low"
        else:
            priority = "medium"

        issue_match = re.search(
            r"((?:xe|phòng|dịch vụ)[^.\n]+?(?:bị|lỗi|hỏng|ẩm mốc)[^.\n]*)",
            user_input,
            re.IGNORECASE
        )
        if issue_match:
            issue_description = issue_match.group(1).strip()
        else:
            issue_description = user_input.strip()

        return {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority
        }

    def _format_catalog_results(self, products: List[Dict[str, Any]]) -> str:
        """Định dạng kết quả tra cứu sản phẩm."""
        if not products:
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của quý khách."
        lines = ["Dưới đây là các sản phẩm phù hợp:"]
        for p in products:
            price_vnd = p.get("price_vnd", 0)
            price_str = f"{price_vnd:,}".replace(",", ".")
            name = p.get("name", "")
            desc = p.get("description", "")
            lines.append(f"- {name}: Giá {price_str} VNĐ. {desc}")
        return "\n".join(lines)

    def _format_ticket_result(self, ticket_info: Dict[str, Any]) -> str:
        """Định dạng kết quả tạo ticket."""
        ticket_id = ticket_info.get("ticket_id", "")
        customer = ticket_info.get("customer_name", "quý khách")
        priority = ticket_info.get("priority", "medium")
        status = ticket_info.get("status", "open")
        return (
            f"Yêu cầu hỗ trợ của khách hàng {customer} đã được ghi nhận thành công.\n"
            f"- Mã phiếu hỗ trợ: {ticket_id}\n"
            f"- Mức độ ưu tiên: {priority}\n"
            f"- Trạng thái: {status}\n"
            f"Đội ngũ kỹ thuật Vingroup sẽ liên hệ xử lý trong thời gian sớm nhất."
        )

    def _format_faq_answer(self, user_input: str) -> str:
        """Định dạng câu trả lời cho các câu hỏi thường gặp (FAQ)."""
        lower = user_input.lower()
        if "bảo hành" in lower and ("pin" in lower or "xe" in lower):
            return (
                "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm (không giới hạn số km tuỳ dòng xe) "
                "hoặc áp dụng theo chính sách bảo hành chính hãng của VinFast. "
                "Quý khách hoàn toàn yên tâm về chất lượng và độ bền của pin khi sử dụng xe điện VinFast."
            )
        return (
            "Xin chào quý khách! Tôi là VinAssistant, trợ lý AI chính thức của Vingroup. "
            "Tôi có thể hỗ trợ quý khách tra cứu thông tin sản phẩm xe điện VinFast, dịch vụ Vinpearl "
            "hoặc tiếp nhận các yêu cầu hỗ trợ kỹ thuật."
        )

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        # Safeguard: Max iterations check
        if self.max_iterations <= 0:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached"
            }

        intents = self._detect_intents(user_input)

        # Lập danh sách công cụ cần thực thi
        planned_actions = []
        if intents["needs_catalog"]:
            planned_actions.append(("search_product_catalog", self._extract_catalog_args(user_input)))
        if intents["needs_ticket"]:
            planned_actions.append(("submit_support_ticket", self._extract_ticket_args(user_input)))

        iteration = 1
        catalog_results = None
        ticket_result = None

        if intents["is_faq"] or not planned_actions:
            # FAQ hoặc phản hồi trực tiếp: 1 iteration
            answer = self._format_faq_answer(user_input)
            self.trace.append({
                "iteration": iteration,
                "thought": "Câu hỏi thuộc dạng thông tin chung / FAQ, không cần tra cứu dữ liệu động hay tạo ticket.",
                "action": None,
                "action_input": None,
                "observation": None,
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # Agent Loop thực thi các công cụ
        action_idx = 0
        while iteration <= self.max_iterations and action_idx < len(planned_actions):
            tool_name, tool_args = planned_actions[action_idx]

            thought = (
                f"Người dùng cần tra cứu sản phẩm danh mục '{tool_args.get('category')}'. Gọi công cụ {tool_name}."
                if tool_name == "search_product_catalog"
                else f"Người dùng báo cáo sự cố hoặc cần hỗ trợ. Gọi công cụ {tool_name}."
            )

            tool_func = TOOL_MAP.get(tool_name)
            observation = tool_func(**tool_args) if tool_func else {"error": f"Tool {tool_name} not found"}

            if tool_name == "search_product_catalog":
                catalog_results = observation
            elif tool_name == "submit_support_ticket":
                ticket_result = observation

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": tool_name,
                "action_input": tool_args,
                "observation": observation
            })

            action_idx += 1
            if action_idx >= len(planned_actions):
                # Hoàn thành tất cả công cụ cần thiết, tổng hợp Final Answer
                answer_parts = []
                if catalog_results is not None:
                    answer_parts.append(self._format_catalog_results(catalog_results))
                if ticket_result is not None:
                    answer_parts.append(self._format_ticket_result(ticket_result))

                final_answer = "\n\n".join(answer_parts)
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            iteration += 1

        # Vượt quá số bước tối đa
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:\n", result["answer"])
    print("\nTrace Log:\n", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
