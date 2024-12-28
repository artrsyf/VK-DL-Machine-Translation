import React, { useState, useEffect } from "react";
import axios from "axios";

const Translator: React.FC = () => {
  const [inputText, setInputText] = useState<string>("");
  const [translatedText, setTranslatedText] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Делаем запрос перевода
  const handleTranslate = async (text: string) => {
    if (!text.trim()) return; // Если текст пустой, не отправляем запрос

    try {
      setIsLoading(true); // Включаем индикатор загрузки
      const response = await axios.post("http://localhost:8000/translate", {
        text: text,
      });
      setTranslatedText(response.data.translated_text); // Сохраняем переведенный текст
    } catch (error) {
      console.error("Ошибка при переводе:", error);
      setTranslatedText("Произошла ошибка при переводе.");
    } finally {
      setIsLoading(false); // Отключаем индикатор загрузки
    }
  };

  // Используем useEffect, чтобы делать запрос при изменении текста
  useEffect(() => {
    // Добавляем задержку перед отправкой запроса
    const timer = setTimeout(() => {
      handleTranslate(inputText);
    }, 500); // Задержка в 500ms

    // Очистка таймера, если текст был изменен до окончания таймера
    return () => clearTimeout(timer);
  }, [inputText]);

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", textAlign: "center" }}>
      <h1>Machine Translation</h1>
      
      {/* Контейнер с флексом для размещения ввода и вывода на одной строке */}
      <div
        style={{
          display: "flex",
          flexDirection: "row", // Размещение элементов на одной строке
          gap: "20px", // Отступ между элементами
          justifyContent: "center", // Выравнивание элементов по центру
          alignItems: "flex-start", // Выравнивание по верхнему краю
        }}
      >
        {/* Поле ввода */}
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Введите текст для перевода"
          style={{
            width: "45%", // Уменьшаем ширину, чтобы оба блока поместились
            height: "100px",
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #ddd",
            backgroundColor: "#f5f5f5",
            resize: "none", // Отключаем возможность изменения размера
            fontSize: "16px",
            boxSizing: "border-box", // Чтобы padding не увеличивал размер
          }}
        />
        
        {/* Вывод перевода */}
        <div
          style={{
            width: "45%", // Уменьшаем ширину, чтобы оба блока поместились
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #ddd",
            backgroundColor: "#f5f5f5",
            fontSize: "16px",
            minHeight: "100px", // Минимальная высота, чтобы блок не сжимался
            display: "flex",
            alignItems: "center", // Выравнивание текста по вертикали
            justifyContent: "center", // Выравнивание текста по горизонтали
            boxSizing: "border-box", // Чтобы padding не увеличивал размер
          }}
        >
          {isLoading ? "Загрузка..." : translatedText || "Здесь появится перевод..."}
        </div>
      </div>
    </div>
  );
};

export default Translator;