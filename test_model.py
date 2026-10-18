"""
المساعد الذكي لنظام العمل السعودي (Saudi Labor Law Assistant)
============================================================
يدعم الاختيار السريع بالأرقام أو كتابة الأسئلة العربية مباشرة.
"""

import os
import sys
import torch

# محاولة استيراد مكتبات التنسيق العربي إن كانت مثبتة
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_BIDI = True
except ImportError:
    HAS_BIDI = False

def fix_rtl(text: str) -> str:
    """إعادة تشكيل وعكس النصوص العربية للعرض الصحيح في الـ Terminal"""
    if HAS_BIDI:
        try:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except Exception:
            return text
    return text

from micro_llm import SadaLLM, CONFIG_10M, generate_story

PRESET_QUESTIONS = {
    "1": "سؤال: كم الحد الأقصى لساعات العمل اليومية في نظام العمل السعودي؟\nالجواب:",
    "2": "سؤال: كم تبلغ فترة التجربة في نظام العمل؟\nالجواب:",
    "3": "سؤال: كيف تحسب مكافأة نهاية الخدمة للعامل؟\nالجواب:",
    "4": "سؤال: كم يستحق العامل إجازة سنوية في نظام العمل السعودي؟\nالجواب:",
    "5": "سؤال: متى يستحق العامل إجازة الزواج والوفاة؟\nالجواب:",
    "6": "المادة الثالثة والخمسون: إذا كان العامل خاضعاً لفترة تجربة",
    "7": "المادة الرابعة والثمانون: إذا انتهت علاقة العمل وجب على صاحب العمل"
}

def main():
    print("\n" + "=" * 75)
    print("      المساعد الذكي لنظام العمل السعودي (SadaLLM - صدى)")
    print("=" * 75)

    model = SadaLLM(CONFIG_10M)
    checkpoint_path = "saudi_labor_law_model.pt"

    if os.path.exists(checkpoint_path):
        print(f"جاري فحص وتحميل أوزان النموذج من '{checkpoint_path}'...")
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print("تم تحميل النموذج بنجاح بنظام الحروف العربية!")
        except Exception:
            print("تم الترقية إلى مرمز الحروف العربية. جاري إعادة تدريب الأوزان بالمرمز الجديد...")
            from micro_llm import train_model
            model = train_model(CONFIG_10M, max_steps=450, batch_size=16)
            torch.save(model.state_dict(), checkpoint_path)
    else:
        print(f"تنبيه: لم يتم العثور على أوزان محفوظة ({checkpoint_path}).")
        print("جاري تدريب النموذج على نصوص نظام العمل السعودي الآن...")
        from micro_llm import train_model
        model = train_model(CONFIG_10M, max_steps=450, batch_size=16)
        torch.save(model.state_dict(), checkpoint_path)

    model.eval()

    current_temp = 0.65
    current_top_k = 15

    while True:
        print("\n" + "-" * 75)
        print("اختر رقم السؤال أو اكتب استفسارك بالعربي:")
        print("  [1] ساعات العمل اليومية والأسبوعية")
        print("  [2] ضوابط وتمديد فترة التجربة")
        print("  [3] كيفية حساب مكافأة نهاية الخدمة")
        print("  [4] رصيد الإجازة السنوية النظامية")
        print("  [5] إجازات المناسبات (الزواج والوفاة والمولود)")
        print("  [6] نص المادة 53 (فترة التجربة)")
        print("  [7] نص المادة 84 (مكافأة نهاية الخدمة)")
        print("  [اكتب أي سؤال مخصص، أو اكتب 'خروج' أو 'exit' للإنهاء]")
        print("-" * 75)

        try:
            user_input = input("\n[اختر رقماً أو اكتب سؤالك]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nتم إنهاء البرنامج.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "خروج", "q"]:
            print("شكراً لك! بالتوفيق لطلابك.")
            break

        # Check if chosen from presets
        if user_input in PRESET_QUESTIONS:
            prompt_formatted = PRESET_QUESTIONS[user_input]
        else:
            prompt_formatted = user_input
            if "سؤال:" in user_input and "الجواب:" not in user_input:
                prompt_formatted = f"{user_input}\nالجواب:"

        print(f"\n[المدخل المرسل للنموذج]:\n{prompt_formatted}")
        print("\n[جاري توليد الإجابة النظامية عبر الذكاء الاصطناعي...]")
        print("-" * 70)

        output = generate_story(
            model,
            prompt=prompt_formatted,
            max_new_tokens=140,
            temperature=current_temp,
            top_k=current_top_k
        )
        print(fix_rtl(output))
        print("-" * 70)

if __name__ == "__main__":
    main()
