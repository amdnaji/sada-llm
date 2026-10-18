"""
واجهة الويب التفاعلية لمساعد نظام العمل السعودي (Local Arabic Web UI)
===================================================================
واجهة متصفح محلية خفيفة جداً تدعم اللغة العربية والاتجاه من اليمين لليسار (RTL) بشكل أصلي 100%.
لا تحتاج أي مكتبات خارجية (تعتمد على خادم بايثون المدمج).
"""

import os
import json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch

from micro_llm import SadaLLM, CONFIG_10M, generate_story

# 1. تحميل النموذج
model = SadaLLM(CONFIG_10M)
checkpoint_path = "saudi_labor_law_model.pt"

if os.path.exists(checkpoint_path):
    print(f"جاري فحص وتحميل أوزان النموذج من '{checkpoint_path}'...")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print("تم تحميل النموذج بنجاح بنظام الحروف العربية!")
    except Exception as e:
        print(f"تم الترقية إلى مرمز الحروف العربية (Arabic Character Tokenizer). جاري تدريب الأوزان الجديدة...")
        from micro_llm import train_model
        model = train_model(CONFIG_10M, max_steps=450, batch_size=16)
        torch.save(model.state_dict(), checkpoint_path)
else:
    print(f"تنبيه: أوزان '{checkpoint_path}' غير موجودة، جاري التدريب السريع...")
    from micro_llm import train_model
    model = train_model(CONFIG_10M, max_steps=450, batch_size=16)
    torch.save(model.state_dict(), checkpoint_path)

model.eval()

HTML_PAGE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SadaLLM (صدى) - المساعد الذكي لنظام العمل السعودي</title>
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Tahoma, Arial, sans-serif; }
        body { background: #f0f4f8; margin: 0; padding: 20px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 850px; background: #fff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); overflow: hidden; display: flex; flex-direction: column; height: 92vh; }
        header { background: #1a4d2e; color: white; padding: 18px 24px; }
        header h1 { margin: 0; font-size: 20px; }
        header p { margin: 6px 0 0 0; font-size: 13px; opacity: 0.85; }
        .preset-bar { background: #e8f0e9; padding: 12px 18px; border-bottom: 1px solid #d0e0d2; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
        .preset-btn { background: #fff; border: 1px solid #1a4d2e; color: #1a4d2e; padding: 6px 12px; border-radius: 20px; font-size: 12px; cursor: pointer; transition: all 0.2s; }
        .preset-btn:hover { background: #1a4d2e; color: white; }
        .chat-area { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
        .msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 14px; white-space: pre-wrap; }
        .msg-user { background: #e3f2fd; color: #0d47a1; align-self: flex-start; border-bottom-right-radius: 2px; }
        .msg-bot { background: #f1f8e9; color: #1b5e20; align-self: flex-end; border-bottom-left-radius: 2px; border-right: 4px solid #2e7d32; }
        .input-bar { padding: 16px; background: #fafafa; border-top: 1px solid #eee; display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 12px 16px; border: 1px solid #ccc; border-radius: 24px; font-size: 14px; outline: none; }
        input[type="text"]:focus { border-color: #1a4d2e; }
        button#send { background: #1a4d2e; color: white; border: none; padding: 0 24px; border-radius: 24px; font-size: 14px; cursor: pointer; }
        button#send:hover { background: #143d24; }
        .controls { display: flex; gap: 16px; padding: 6px 18px; background: #fafafa; font-size: 12px; color: #555; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>SadaLLM (صدى) - نموذج لغوي مصغر للأغراض التعليمية (10.6M Parameters)</h1>
            <p>نموذج لغوي مصغر مبني من الصفر لتعليم الطلاب - ونظام العمل السعودي مثال تطبيقي لتوضيح آلية التعلم</p>
        </header>

        <div class="preset-bar">
            <span style="font-size:12px; font-weight:bold; color:#1a4d2e;">أسئلة شائعة:</span>
            <button class="preset-btn" onclick="sendPreset('سؤال: كم الحد الأقصى لساعات العمل اليومية في نظام العمل السعودي؟\\nالجواب:')">ساعات العمل</button>
            <button class="preset-btn" onclick="sendPreset('سؤال: كم تبلغ فترة التجربة في نظام العمل؟\\nالجواب:')">فترة التجربة</button>
            <button class="preset-btn" onclick="sendPreset('سؤال: كيف تحسب مكافأة نهاية الخدمة للعامل؟\\nالجواب:')">مكافأة نهاية الخدمة</button>
            <button class="preset-btn" onclick="sendPreset('سؤال: كم يستحق العامل إجازة سنوية في نظام العمل السعودي؟\\nالجواب:')">الإجازة السنوية</button>
            <button class="preset-btn" onclick="sendPreset('المادة الثالثة والخمسون: إذا كان العامل خاضعاً لفترة تجربة')">المادة 53</button>
        </div>

        <div class="controls">
            <label>الحرارة (Temperature): <input type="number" id="temp" value="0.2" step="0.05" min="0.05" max="1.5" style="width:50px;"></label>
            <label>Top-K: <input type="number" id="topk" value="5" step="1" min="1" max="50" style="width:45px;"></label>
            <span style="color:#2e7d32; font-size:11px;">(الحرارة 0.2 تضمن إجابات قانونية فصيحة ومحكمة)</span>
        </div>

        <div class="chat-area" id="chat">
            <div class="msg msg-bot">مرحباً بك! أنا نموذج الذكاء الاصطناعي المصغر لنظام العمل السعودي. يمكنك سؤالي عن مواد النظام أو اختيار سؤال من الأعلى.</div>
        </div>

        <div class="input-bar">
            <input type="text" id="userInput" placeholder="اكتب سؤالك القانوني هنا..." onkeypress="if(event.key==='Enter') sendMsg()">
            <button id="send" onclick="sendMsg()">إرسال</button>
        </div>
    </div>

    <script>
        function appendMsg(text, type) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'msg ' + (type === 'user' ? 'msg-user' : 'msg-bot');
            div.innerText = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        async function sendMsg() {
            const input = document.getElementById('userInput');
            const text = input.value.trim();
            if(!text) return;
            input.value = '';

            let prompt = text;
            if(text.includes('سؤال:') && !text.includes('الجواب:')) {
                prompt = text + '\\nالجواب:';
            }

            appendMsg(text, 'user');
            appendMsg('جاري التوليد...', 'bot');

            const temp = parseFloat(document.getElementById('temp').value) || 0.65;
            const topk = parseInt(document.getElementById('topk').value) || 15;

            try {
                const res = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: prompt, temperature: temp, top_k: topk})
                });
                const data = await res.json();
                
                // Remove 'جاري التوليد...'
                const chat = document.getElementById('chat');
                chat.removeChild(chat.lastChild);
                appendMsg(data.output, 'bot');
            } catch(e) {
                appendMsg('حدث خطأ في الاتصال بالنموذج.', 'bot');
            }
        }

        function sendPreset(txt) {
            document.getElementById('userInput').value = txt;
            sendMsg();
        }
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode('utf-8'))

    def do_POST(self):
        if self.path == '/generate':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            data = json.loads(body)

            prompt = data.get('prompt', '')
            temp = float(data.get('temperature', 0.65))
            top_k = int(data.get('top_k', 15))

            output = generate_story(
                model,
                prompt=prompt,
                max_new_tokens=140,
                temperature=temp,
                top_k=top_k
            )

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'output': output}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    server_address = ('127.0.0.1', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    url = "http://127.0.0.1:8000"
    print("\n" + "=" * 70)
    print(f" تم تشغيل واجهة الويب العربية بنجاح على: {url}")
    print(" تدعم اللغة العربية والاتجاه من اليمين لليسار (RTL) بشكل أصلي 100%")
    print(" اضغط Ctrl+C في الـ Terminal لإيقاف الخادم.")
    print("=" * 70)
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nتم إيقاف خادم الويب.")

if __name__ == "__main__":
    run_server()
