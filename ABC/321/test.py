def generate_numbers(min_digits, max_digits):
    result = []
    
    for num_digits in range(min_digits, max_digits + 1):
        for i in range(10 ** (num_digits - 1), 10 ** num_digits):
            num_str = str(i)
            
            # 各桁の数字を取得
            digits = [int(digit) for digit in num_str]
            
            # 条件を満たすかをチェック
            if all(digits[j] < digits[j+1] for j in range(num_digits - 1)):
                result.append(i)
    
    return result

# 1桁から10桁までの数値を生成
min_digits = 1
max_digits = 10
numbers = generate_numbers(min_digits, max_digits)

# 結果を表示
print(numbers)
