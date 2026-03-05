from PIL import Image

def is_color_in_range(pixel, target_color, tolerance):
    """Проверяем, попадает ли цвет пикселя в заданный диапазон."""
    return all(abs(pixel[i] - target_color[i]) <= tolerance for i in range(3))

def chroma_key(image1_path, image2_path, target_color, replacement_color, tolerance=30):
    # Загружаем изображения
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    # Убедимся, что размеры изображений совпадают
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    # Создаем новое изображение для результата
    result_image = Image.new('RGB', image1.size)

    # Обрабатываем каждый пиксель
    for x in range(image1.width):
        for y in range(image1.height):
            pixel1 = image1.getpixel((x, y))
            
            # Проверяем, является ли пиксель целевым цветом
            if is_color_in_range(pixel1, target_color, tolerance):
                # Если да, берем пиксель из второго изображения
                pixel2 = image2.getpixel((x, y))
                result_image.putpixel((x, y), pixel2)
            else:
                # Если нет, оставляем оригинальный пиксель
                result_image.putpixel((x, y), pixel1)

    return result_image

# Пример использования
if __name__ == "__main__":
    target_color = (0, 255, 0)  # Зеленый цвет
    replacement_color = (255, 0, 0)  # Здесь можно указать нужный цвет для замены

    result_image = chroma_key('image1.jpg', 'image2.jpg', target_color, replacement_color)
    result_image.show()  # Показываем результат
    result_image.save('result_image.jpg')  # Сохраняем результат
