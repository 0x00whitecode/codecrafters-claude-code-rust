use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    // conversation memory
    let mut messages = vec![json!({
        "role": "user",
        "content": args.prompt
    })];

    loop {
        let response: Value = client
            .chat()
            .create_byot(json!({
                "model": "anthropic/claude-haiku-4.5",
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Read and return the contents of a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path to the file to read"
                                    }
                                },
                                "required": ["file_path"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "description": "Write content to a file",
                            "parameters": {
                                "type": "object",
                                "required": ["file_path", "content"],
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path of the file to write to"
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The content to write to the file"
                                    }
                                }
                            }
                        }
                    }
                ]
            }))
            .await?;

        let message = &response["choices"][0]["message"];

        // store assistant message
        messages.push(message.clone());

        // handle tool calls
        if let Some(tool_calls) = message["tool_calls"].as_array() {
            for tool_call in tool_calls {
                let function_name = tool_call["function"]["name"]
                    .as_str()
                    .unwrap_or("");

                let arguments_str = tool_call["function"]["arguments"]
                    .as_str()
                    .unwrap_or("{}");

                let arguments: Value = serde_json::from_str(arguments_str)?;

                let result = match function_name {
                    "Read" => {
                        let file_path = arguments["file_path"]
                            .as_str()
                            .unwrap_or("");

                        match std::fs::read_to_string(file_path) {
                            Ok(content) => content,
                            Err(e) => format!("Error reading file: {}", e),
                        }
                    }

                    "Write" => {
                        let file_path = arguments["file_path"]
                            .as_str()
                            .unwrap_or("");

                        let content = arguments["content"]
                            .as_str()
                            .unwrap_or("");

                        match std::fs::write(file_path, content) {
                            Ok(_) => format!("Successfully wrote to {}", file_path),
                            Err(e) => format!("Error writing file: {}", e),
                        }
                    }

                    _ => "Unknown tool".to_string(),
                };

                // send tool result back to model
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result
                }));
            }

            // let model continue after tool execution
            continue;
        }

        // final response from model
        if let Some(content) = message["content"].as_str() {
            println!("{}", content);
        }

        break;
    }

    Ok(())
}