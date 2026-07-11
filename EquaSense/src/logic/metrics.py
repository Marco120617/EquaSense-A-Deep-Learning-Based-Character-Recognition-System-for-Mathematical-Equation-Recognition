def normalize_equation(text):
    if text is None:
        return ""
    text = text.lower()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    return text

def levenshtein_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def calculate_cer(predicted, expected):
    predicted = normalize_equation(predicted)
    expected = normalize_equation(expected)
    if len(expected) == 0:
        return 0.0
    return levenshtein_distance(predicted, expected) / len(expected)

def calculate_accuracy(predicted, expected):
    cer = calculate_cer(predicted, expected)
    return max(0.0, (1.0 - cer) * 100.0)