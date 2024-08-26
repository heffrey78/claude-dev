import { Anthropic } from "@anthropic-ai/sdk"
import { ApiHandler, withoutImageData } from "."
import {
	ApiHandlerOptions,
	ModelInfo,
	ollamaDefaultModelId,
	ollamaModelId,
	ollamaModels,
} from "../shared/api"
import { Ollama, Tool } from "ollama"
import defaultShell from "default-shell"
import path from "path"
import os from "os"
import osName from "os-name"
import * as vscode from "vscode"

const cwd = vscode.workspace.workspaceFolders?.map((folder) => folder.uri.fsPath).at(0) ?? path.join(os.homedir(), "Desktop");
const ollamaSystemPrompt =
	() => `You are a coding assistant integrated into VS Code, powered by the llama3.1 7B language model. Your primary function is to assist users with coding tasks, provide explanations, and offer suggestions for code improvement.

CAPABILITIES:
- Perform any task that can be executed from a commandline focusing on coding tasks
- Ability to determine what tools may be called in order to complete your task
- Analyze and understand code in various programming languages
- Provide explanations for code functionality and suggest improvements
- Assist with debugging and error resolution
- Offer insights on best practices and design patterns

RULES:
- Work within the context of the currently open file in VS Code
- Provide complete code snippets when suggesting changes or additions
- Use markdown and appropriate language tags for code blocks
- Avoid asking unnecessary follow-up questions; infer from context and workspace

OBJECTIVE:
1. Analyze the user's request in the context of the current VS Code workspace
2. Provide a clear and concise response that directly addresses the user's needs
3. If code changes are suggested, present them in a format that can be easily copied and pasted
4. Conclude your response with a summary of the changes or advice provided

SYSTEM INFORMATION

Operating System: ${osName()}
Default Shell: ${defaultShell}
Home Directory: ${os.homedir()}
Current Working Directory: ${cwd}

TOOL EXAMPLES
read_file(path="data.txt") # Reads the contents of data.txt
write_to_file(path="output.txt", content="Hello, world!") # Writes "Hello, world!" to output.txt
ask_followup_question(question="What is your name?") # Asks the user their name
attempt_completion(result="The task has been completed successfully.") # Presents the result to the user

Remember to tailor your responses to the specific capabilities of the llama3.1 7B model, focusing on providing accurate and helpful coding assistance within the limitations of the model's knowledge and reasoning abilities.`
const ollamaTools: Tool[] = [
	{
		type: "function",
		function: {
			name: "execute_command",
			description: `Execute a CLI command on the system. Use this when you need to perform system operations or run specific commands to accomplish any step in the user's task. You must tailor your command to the user's system and provide a clear explanation of what the command does. Prefer to execute complex CLI commands over creating executable scripts, as they are more flexible and easier to run. Commands will be executed in the current working directory: ${cwd}`,
			parameters: {
				type: "object",
				properties: {
					command: {
						type: "string",
						description:
							"The CLI command to execute. This should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions.",
					},
				},
				required: ["command"],
			}
		}
	},
	{
		type: "function",
		function: {
			name: "list_files_top_level",
			description:
				"List all files and directories at the top level of the specified directory. This should only be used for generic directories you don't necessarily need the nested structure of, like the Desktop.",
			parameters: {
				type: "object",
				properties: {
					path: {
						type: "string",
						description: `The path of the directory to list contents for (relative to the current working directory ${cwd})`,
					},
				},
				required: ["path"],
			},
		}
	},
	{
		type: "function",
		function: {
			name: "list_files_recursive",
			description:
				"Recursively list all files and directories within the specified directory. This provides a comprehensive view of the project structure, and can guide decision-making on which files to process or explore further.",
			parameters: {
				type: "object",
				properties: {
					path: {
						type: "string",
						description: `The path of the directory to recursively list contents for (relative to the current working directory ${cwd})`,
					},
				},
				required: ["path"],
			},
		}
	},
	{
		type: "function",
		function: {
			name: "view_source_code_definitions_top_level",
			description:
				"Parse all source code files at the top level of the specified directory to extract names of key elements like classes and functions. This tool provides insights into the codebase structure and important constructs, encapsulating high-level concepts and relationships that are crucial for understanding the overall architecture.",
			parameters: {
				type: "object",
				properties: {
					path: {
						type: "string",
						description: `The path of the directory (relative to the current working directory ${cwd}) to parse top level source code files for to view their definitions`,
					},
				},
				required: ["path"],
			},
		}
	},
	{
		type: "function",
		function: {
			name: "read_file",
			description:
				"Read the contents of a file at the specified path. Use this when you need to examine the contents of an existing file, for example to analyze code, review text files, or extract information from configuration files. Be aware that this tool may not be suitable for very large files or binary files, as it returns the raw content as a string.",
			parameters: {
				type: "object",
				properties: {
					path: {
						type: "string",
						description: `The path of the file to read (relative to the current working directory ${cwd})`,
					},
				},
				required: ["path"],
			},
		}
	},
	{
		type: "function",
		function: {
			name: "write_to_file",
			description:
				"Write content to a file at the specified path. If the file exists, it will be completely overwritten with the provided content (so do NOT omit unmodified sections). If the file doesn't exist, it will be created. This tool will automatically create any directories needed to write the file.",
			parameters: {
				type: "object",
				properties: {
					path: {
						type: "string",
						description: `The path of the file to write to (relative to the current working directory ${cwd})`,
					},
					content: {
						type: "string",
						description:
							"The full content to write to the file. Must be the full intended content of the file, without any omission or truncation.",
					},
				},
				required: ["path", "content"],
			},
		}
	},
	{
		type: "function",
		function: {
			name: "ask_followup_question",
			description:
				"Ask the user a question to gather additional information needed to complete the task. This tool should be used when you encounter ambiguities, need clarification, or require more details to proceed effectively. It allows for interactive problem-solving by enabling direct communication with the user. Use this tool judiciously to maintain a balance between gathering necessary information and avoiding excessive back-and-forth.",
			parameters: {
				type: "object",
				properties: {
					question: {
						type: "string",
						description:
							"The question to ask the user. This should be a clear, specific question that addresses the information you need.",
					},
				},
				required: ["question"],
			},
		}
	},
	{
		type: "function",
		function: {
			name: "attempt_completion",
			description:
				"Once you've completed the task, use this tool to present the result to the user. They may respond with feedback if they are not satisfied with the result, which you can use to make improvements and try again.",
			parameters: {
				type: "object",
				properties: {
					command: {
						type: "string",
						description:
							"The CLI command to execute to show a live demo of the result to the user. For example, use 'open index.html' to display a created website. This should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions.",
					},
					result: {
						type: "string",
						description:
							"The result of the task. Formulate this result in a way that is final and does not require further input from the user. Don't end your result with questions or offers for further assistance.",
					},
				},
				required: ["result"],
			},
		}
	},
]


type AnthropicSchemaProperty = {
	type: string;
	description?: string;
	enum?: string[];
};


export class OllamaHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private ollama: Ollama

	constructor(options: ApiHandlerOptions) {
		this.options = options;
		this.ollama = new Ollama({ host: "http://127.0.0.1:11434" });
	}

	async createMessage(
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
		tools: Anthropic.Messages.Tool[]
	): Promise<Anthropic.Messages.Message> {
		const modelId = this.getModel().id
		// Convert Anthropic messages to Ollama format
		const ollamaMessages = this.convertToOllamaMessages(ollamaSystemPrompt.toString(), messages)

		const response = await this.ollama.chat({
			model: modelId,
			messages: ollamaMessages,
			stream: false,
			tools: ollamaTools
		});

		// Convert Ollama response to Anthropic format
		const anthropicMessage: Anthropic.Messages.Message = {
			id: `ollama-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
			type: "message",
			role: "assistant",
			content: [
				{
					type: "text",
					text: response.message.content,
				},
			],
			model: this.getModel().id,
			stop_reason: response.message.tool_calls ? "tool_use" : "end_turn",
			stop_sequence: null,
			usage: {
				input_tokens: response.prompt_eval_count,
				output_tokens: response.eval_count,
			},
		}

		if (response.message.tool_calls && response.message.tool_calls.length > 0) {
			anthropicMessage.content.push(
				...this.convertOllamaToolToAnthropicToolUseBlock(response.message)
			);
		}

		return anthropicMessage
	}

	convertOllamaToolToAnthropicToolUseBlock(ollamaMessage: any): Anthropic.ToolUseBlock[] {
		if (!ollamaMessage.tool_calls || ollamaMessage.tool_calls.length === 0) {
			return [];
		}

		return ollamaMessage.tool_calls.map((toolCall: any): Anthropic.ToolUseBlock => {
			let input: any;

			if (typeof toolCall.function.arguments === 'string') {
				try {
					input = JSON.parse(toolCall.function.arguments);
				} catch (error) {
					console.error("Failed to parse tool arguments:", error);
					input = {}; // Fallback to empty object if parsing fails
				}
			} else if (typeof toolCall.function.arguments === 'object' && toolCall.arguments !== null) {
				input = toolCall.function.arguments; // Use as-is if it's already an object
			} else {
				input = {}; // Fallback to empty object for other cases
			}

			return {
				type: "tool_use",
				id: toolCall.function.id, // do we need to set a random number here?
				name: toolCall.function.name,
				input: input,
			};
		});
	}


	convertToOllamaMessages(
		systemPrompt: string,
		anthropicMessages: Anthropic.Messages.MessageParam[]
	): { role: string; content: string }[] {
		const ollamaMessages: { role: string; content: string }[] = [
			{ role: "system", content: systemPrompt },
		]

		for (const message of anthropicMessages) {
			if (typeof message.content === "string") {
				ollamaMessages.push({ role: message.role, content: message.content })
			} else {
				const content = message.content
					.map(part => {
						if (part.type === "text") { return part.text }
						if (part.type === "image") { return "[Image]" } // Ollama doesn't support image inputs
						if (part.type === "tool_use") { return `[Tool Use: ${part.name}]` }
						if (part.type === "tool_result") { return `[Tool Result: ${part.tool_use_id}]` }
						return ""
					})
					.join("\n")
				ollamaMessages.push({ role: message.role, content })
			}
		}

		return ollamaMessages
	}

	createUserReadableRequest(
		userContent: Array<
			| Anthropic.TextBlockParam
			| Anthropic.ImageBlockParam
			| Anthropic.ToolUseBlockParam
			| Anthropic.ToolResultBlockParam
		>
	): any {
		return {
			model: this.getModel().id,
			messages: [
				{ role: "system", content: "(see SYSTEM_PROMPT in src/ClaudeDev.ts)" },
				{
					role: "user", content: withoutImageData(userContent).map(part => {
						if (part.type === "text") { return part.text }
						if (part.type === "image") { return "[Image]" }
						if (part.type === "tool_use") { return `[Tool Use: ${part.name}]` }
						if (part.type === "tool_result") { return `[Tool Result: ${part.tool_use_id}]` }
						return ""
					}).join("\n")
				},
			],
		}
	}

	getModel(): { id: ollamaModelId; info: ModelInfo } {
		const modelId = this.options.apiModelId
		if (modelId && modelId in ollamaModels) {
			const id = modelId as ollamaModelId
			return { id, info: ollamaModels[id] }
		}
		return { id: ollamaDefaultModelId, info: ollamaModels[ollamaDefaultModelId] }
	}
}