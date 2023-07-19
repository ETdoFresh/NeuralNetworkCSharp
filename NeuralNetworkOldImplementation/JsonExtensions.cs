using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using NeuralNetwork.MatrixModels;

namespace NeuralNetwork
{
    public static class JsonExtensions
    {
        private static JsonSerializerOptions _jsonSerializerOptions = new()
            {Converters = {new JsonModelConverter()}, PropertyNameCaseInsensitive = true,};

        public static T? ToObject<T>(this JsonElement element)
        {
            var json = element.GetRawText();
            return JsonSerializer.Deserialize<T>(json, _jsonSerializerOptions);
        }

        public static T? ToObject<T>(this JsonDocument document)
        {
            var json = document.RootElement.GetRawText();
            return JsonSerializer.Deserialize<T>(json, _jsonSerializerOptions);
        }

        public static JsonElement AddProperty<T>(this JsonElement jsonElement, string key, T value)
        {
            using MemoryStream memoryStream = new();
            using (Utf8JsonWriter utf8JsonWriter = new(memoryStream))
            {
                utf8JsonWriter.WriteStartObject();
                var wroteValue = false;
                if (jsonElement.ValueKind == JsonValueKind.Object)
                {
                    foreach (var element in jsonElement.EnumerateObject())
                    {
                        if (element.Name == key)
                        {
                            utf8JsonWriter.WritePropertyName(element.Name);
                            WriteValue(utf8JsonWriter, value);
                            wroteValue = true;
                        }
                        else
                        {
                            element.WriteTo(utf8JsonWriter);
                        }
                    }
                }

                if (!wroteValue)
                {
                    utf8JsonWriter.WritePropertyName(key);
                    WriteValue(utf8JsonWriter, value);
                }

                utf8JsonWriter.WriteEndObject();
            }

            var resultJson = Encoding.UTF8.GetString(memoryStream.ToArray());
            return JsonDocument.Parse(resultJson).RootElement;
        }

        private static void WriteValue<T>(Utf8JsonWriter utf8JsonWriter, T value)
        {
            switch (value)
            {
                case int intValue:
                    utf8JsonWriter.WriteNumberValue(intValue);
                    break;
                case float floatValue:
                    utf8JsonWriter.WriteNumberValue(floatValue);
                    break;
                case double doubleValue:
                    utf8JsonWriter.WriteNumberValue(doubleValue);
                    break;
                case bool boolValue:
                    utf8JsonWriter.WriteBooleanValue(boolValue);
                    break;
                case DateTime dateTimeValue:
                    utf8JsonWriter.WriteStringValue(dateTimeValue.ToString("O"));
                    break;
                case JsonElement jsonElementValue:
                    if (jsonElementValue.ValueKind == JsonValueKind.Undefined)
                        jsonElementValue = jsonElementValue.SetValue<string>(null);
                    jsonElementValue.WriteTo(utf8JsonWriter);
                    break;
                default:
                    utf8JsonWriter.WriteStringValue(value.ToString());
                    break;
            }
        }

        public static JsonElement AddArrayItem(this JsonElement jsonElement, JsonElement item)
        {
            using MemoryStream memoryStream = new();
            using (Utf8JsonWriter utf8JsonWriter = new(memoryStream))
            {
                utf8JsonWriter.WriteStartArray();

                if (jsonElement.ValueKind == JsonValueKind.Array)
                    foreach (var element in jsonElement.EnumerateArray())
                        element.WriteTo(utf8JsonWriter);

                item.WriteTo(utf8JsonWriter);
                utf8JsonWriter.WriteEndArray();
                ;
            }

            var resultJson = Encoding.UTF8.GetString(memoryStream.ToArray());
            return JsonDocument.Parse(resultJson).RootElement;
        }

        public static JsonElement AddArrayItems(this JsonElement jsonElement, IEnumerable<JsonElement> items)
        {
            using MemoryStream memoryStream = new();
            using (Utf8JsonWriter utf8JsonWriter = new(memoryStream))
            {
                utf8JsonWriter.WriteStartArray();

                if (jsonElement.ValueKind == JsonValueKind.Array)
                    foreach (var element in jsonElement.EnumerateArray())
                        element.WriteTo(utf8JsonWriter);

                foreach (var item in items)
                    item.WriteTo(utf8JsonWriter);

                utf8JsonWriter.WriteEndArray();
                ;
            }

            var resultJson = Encoding.UTF8.GetString(memoryStream.ToArray());
            return JsonDocument.Parse(resultJson).RootElement;
        }

        public static int Count(this JsonElement jsonElement) => jsonElement.GetArrayLength();

        public static IEnumerable<JsonElement> GetEnumerable(this JsonElement? jsonElement)
        {
            if (jsonElement is not {ValueKind: JsonValueKind.Array}) yield break;
            for (var i = 0; i < jsonElement.Value.GetArrayLength(); i++)
                yield return jsonElement.Value[i];
        }

        public static JsonElement? Get(this JsonElement jsonElement, string key)
        {
            if (jsonElement.ValueKind != JsonValueKind.Object) return null;
            if (jsonElement.TryGetProperty(key, out var value)) return value;
            return null;
        }

        public static JsonElement? Get(this JsonElement jsonElement, int index)
        {
            if (jsonElement.ValueKind != JsonValueKind.Array) return null;
            if (0 <= index && index < jsonElement.GetArrayLength())
                return jsonElement[index];
            return null;
        }

        public static JsonElement SetValue<T>(this JsonElement jsonElement, T? value)
        {
            var output = value?.ToString() ?? "null";
            if (value is string) output = $"\"{JsonEncodedText.Encode(output)}\"";
            if (value is bool) output = output.ToLower();
            return JsonDocument.Parse(output).RootElement;
        }
    }
}