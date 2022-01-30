using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NeuralNetwork.MatrixModels
{
    public class JsonModelConverter : JsonConverter<Model>
    {
        public override bool CanConvert(Type typeToConvert)
        {
            return typeof(Model).IsAssignableFrom(typeToConvert);
        }

        public override void Write(Utf8JsonWriter writer, Model value, JsonSerializerOptions options)
        {
            throw new NotImplementedException();
        }

        public override Model? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            var jsonElement = Read(ref reader, typeToConvert, options, new JsonElement());
            var json = jsonElement.GetRawText();
            var type = jsonElement.Get("ModelType")?.GetString();
            return type switch
            {
                "SequentialModel" => JsonSerializer.Deserialize<SequentialModel>(json),
                //"ConvolutionalModel" => JsonSerializer.Deserialize<ConvolutionalModel>(json),
                _ => throw new NotImplementedException()
            };
        }
        
        private JsonElement Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options,
            JsonElement jsonElement)
        {
            if (reader.TokenType == JsonTokenType.StartObject)
            {
                while (reader.Read())
                {
                    if (reader.TokenType == JsonTokenType.EndObject)
                        return jsonElement;

                    if (reader.TokenType != JsonTokenType.PropertyName)
                        throw new JsonException();

                    string key = reader.GetString();
                    reader.Read();
                    jsonElement =
                        jsonElement.AddProperty(key, Read(ref reader, typeToConvert, options, new JsonElement()));
                }
            }
            else if (reader.TokenType == JsonTokenType.StartArray)
            {
                while (reader.Read())
                {
                    if (reader.TokenType == JsonTokenType.EndArray)
                        return jsonElement;

                    jsonElement = jsonElement.AddArrayItem(Read(ref reader, typeToConvert, options, new JsonElement()));
                }
            }
            else if (reader.TokenType == JsonTokenType.False)
            {
                return new JsonElement().SetValue(reader.GetBoolean());
            }
            else if (reader.TokenType == JsonTokenType.True)
            {
                return new JsonElement().SetValue(reader.GetBoolean());
            }
            else if (reader.TokenType == JsonTokenType.Comment)
            {
                return Read(ref reader, typeToConvert, options, jsonElement);
            }
            else if (reader.TokenType == JsonTokenType.Number)
            {
                return new JsonElement().SetValue(reader.GetDouble());
            }
            else if (reader.TokenType == JsonTokenType.String)
            {
                return new JsonElement().SetValue(reader.GetString());
            }
            else if (reader.TokenType == JsonTokenType.None)
            {
                return new JsonElement().SetValue<object>(null);
            }
            else if (reader.TokenType == JsonTokenType.Null)
            {
                return new JsonElement().SetValue<object>(null);
            }

            throw new JsonException();
        }
    }
}