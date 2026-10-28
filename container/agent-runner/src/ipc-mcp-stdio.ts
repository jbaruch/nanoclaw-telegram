/**
 * Stdio MCP Server for NanoClaw
 * Standalone process that agent teams subagents can inherit.
 * Reads context from environment variables, writes IPC files for the host.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import fs from 'fs';
import path from 'path';
import { scheduleInputError } from './schedule-input.js';
import { readTaskSnapshot, TaskSnapshotError } from './task-snapshot.js';
import { isFileSystemError } from './runtime-errors.js';

const IPC_DIR = '/workspace/ipc';
const MESSAGES_DIR = path.join(IPC_DIR, 'messages');
const TASKS_DIR = path.join(IPC_DIR, 'tasks');

// Context from environment variables (set by the agent runner)
const chatJid = process.env.NANOCLAW_CHAT_JID!;
const groupFolder = process.env.NANOCLAW_GROUP_FOLDER!;
const isMain = process.env.NANOCLAW_IS_MAIN === '1';

function writeIpcFile(dir: string, data: object): string {
  fs.mkdirSync(IPC_DIR, { recursive: true, mode: 0o755 });
  const root = fs.openSync(
    IPC_DIR,
    fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_DIRECTORY,
  );
  using _rootLease = { [Symbol.dispose]: () => fs.closeSync(root) };
  // The native agent runs in Linux. Rename through held directory capabilities,
  // so a concurrent host capture cannot redirect the durability acknowledgment.
  const source = `/proc/self/fd/${root}`;
  const filename = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}.json`;
  const tempPath = `${source}/.${filename}.tmp`;
  const fd = fs.openSync(
    tempPath,
    fs.constants.O_WRONLY |
      fs.constants.O_CREAT |
      fs.constants.O_EXCL |
      fs.constants.O_NOFOLLOW,
    0o600,
  );
  using _fileLease = { [Symbol.dispose]: () => fs.closeSync(fd) };
  fs.writeFileSync(fd, JSON.stringify(data, null, 2));
  fs.fsyncSync(fd);
  const identity = fs.fstatSync(fd);
  for (let attempt = 0; attempt < 3; attempt++) {
    let published = false;
    try {
      (() => {
        fs.mkdirSync(dir, { recursive: true, mode: 0o755 });
        const destination = fs.openSync(
          dir,
          fs.constants.O_RDONLY |
            fs.constants.O_NOFOLLOW |
            fs.constants.O_DIRECTORY,
        );
        using _destinationLease = {
          [Symbol.dispose]: () => fs.closeSync(destination),
        };
        fs.renameSync(tempPath, `/proc/self/fd/${destination}/${filename}`);
        published = true;
        fs.fsyncSync(destination);
        fs.fsyncSync(root);
      })();
      return filename;
    } catch (error) {
      // Retry only a known pre-publication missing directory, never an uncertain
      // rename/flush/cleanup outcome after the request has entered host custody.
      if (
        published ||
        !(error instanceof Error) ||
        !('code' in error) ||
        error.code !== 'ENOENT' ||
        attempt === 2
      )
        throw error;
      const stat = fs.lstatSync(tempPath);
      if (
        !stat.isFile() ||
        stat.isSymbolicLink() ||
        stat.nlink !== 1 ||
        stat.dev !== identity.dev ||
        stat.ino !== identity.ino
      )
        throw new Error(
          'IPC unpublished request identity changed; preserve its evidence.',
          { cause: error },
        );
    }
  }
  throw new Error('IPC publication did not complete.');
}

const server = new McpServer({
  name: 'nanoclaw',
  version: '1.0.0',
});

server.tool(
  'send_message',
  "Queue a message for host delivery to the user or group while you're still running. This tool confirms queuing, not channel delivery. Use this for progress updates or to queue multiple messages.",
  {
    text: z.string().describe('The message text to send'),
    sender: z
      .string()
      .optional()
      .describe(
        'Your role/identity name (e.g. "Researcher"). When set, messages appear from a dedicated bot in Telegram.',
      ),
  },
  async (args) => {
    const data: Record<string, string | undefined> = {
      type: 'message',
      chatJid,
      text: args.text,
      sender: args.sender || undefined,
      groupFolder,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(MESSAGES_DIR, data);

    return { content: [{ type: 'text' as const, text: 'Message queued.' }] };
  },
);

server.tool(
  'react_to_message',
  'React to a message with an emoji. Use to acknowledge, approve, or express sentiment without sending a full text reply.',
  {
    messageId: z
      .string()
      .optional()
      .describe(
        'Message ID to react to. If omitted, reacts to the most recent message.',
      ),
    emoji: z
      .string()
      .describe('The emoji to react with (e.g., "👍", "🔥", "✅", "❤️")'),
  },
  async (args) => {
    const data: Record<string, string | undefined> = {
      type: 'react_to_message',
      chatJid,
      messageId: args.messageId || undefined,
      emoji: args.emoji,
      groupFolder,
      timestamp: new Date().toISOString(),
    };
    writeIpcFile(MESSAGES_DIR, data);
    return {
      content: [{ type: 'text' as const, text: `Reacted with ${args.emoji}` }],
    };
  },
);

server.tool(
  'schedule_task',
  `Schedule a recurring or one-time task. The task will run as a full agent with access to all tools. Returns the task ID for future reference. To modify an existing task, use update_task instead.

CONTEXT MODE - Choose based on task type:
\u2022 "group": Task runs in the group's conversation context, with access to chat history. Use for tasks that need context about ongoing discussions, user preferences, or recent interactions.
\u2022 "isolated": Task runs in a fresh session with no conversation history. Use for independent tasks that don't need prior context. When using isolated mode, include all necessary context in the prompt itself.

If unsure which mode to use, you can ask the user. Examples:
- "Remind me about our discussion" \u2192 group (needs conversation context)
- "Check the weather every morning" \u2192 isolated (self-contained task)
- "Follow up on my request" \u2192 group (needs to know what was requested)
- "Generate a daily report" \u2192 isolated (just needs instructions in prompt)

MESSAGING BEHAVIOR - The task agent's output is sent to the user or group. It can also use send_message to queue additional messages for host delivery, or wrap output in <internal> tags to suppress it. Include guidance in the prompt about whether the agent should:
\u2022 Always send a message (e.g., reminders, daily briefings)
\u2022 Only send a message when there's something to report (e.g., "notify me if...")
\u2022 Never send a message (background maintenance tasks)

SCHEDULE VALUE FORMAT (timestamps without a suffix use LOCAL timezone):
\u2022 cron: Standard cron expression (e.g., "*/5 * * * *" for every 5 minutes, "0 9 * * *" for daily at 9am LOCAL time)
\u2022 interval: Milliseconds between runs (e.g., "300000" for 5 minutes, "3600000" for 1 hour)
\u2022 once: ISO timestamp with seconds (e.g., "2026-02-01T15:30:00"). Offset-free values use configured local time; Z or a numeric offset is allowed and required for ambiguous/nonexistent DST wall times.`,
  {
    prompt: z
      .string()
      .describe(
        'What the agent should do when the task runs. For isolated mode, include all necessary context here.',
      ),
    schedule_type: z
      .enum(['cron', 'interval', 'once'])
      .describe(
        'cron=recurring at specific times, interval=recurring every N ms, once=run once at specific time',
      ),
    schedule_value: z
      .string()
      .describe(
        'cron: "*/5 * * * *" | interval: milliseconds like "300000" | once: local timestamp like "2026-02-01T15:30:00" (optional Z or numeric timezone offset)',
      ),
    context_mode: z
      .enum(['group', 'isolated'])
      .default('group')
      .describe(
        'group=runs with chat history and memory, isolated=fresh session (include context in prompt)',
      ),
    target_group_jid: z
      .string()
      .optional()
      .describe(
        '(Main group only) JID of the group to schedule the task for. Defaults to the current group.',
      ),
    script: z
      .string()
      .optional()
      .describe(
        'Optional bash script to run before waking the agent. Script must output JSON on the last line of stdout: { "wakeAgent": boolean, "data"?: any }. If wakeAgent is false, the agent is not called. Test your script with bash -c "..." before scheduling.',
      ),
  },
  async (args) => {
    const validationError = scheduleInputError(
      args.schedule_type,
      args.schedule_value,
    );
    if (validationError)
      return {
        content: [{ type: 'text' as const, text: validationError }],
        isError: true,
      };

    // Non-main groups can only schedule for themselves
    const targetJid =
      isMain && args.target_group_jid ? args.target_group_jid : chatJid;

    const taskId = `task-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const data = {
      type: 'schedule_task',
      taskId,
      prompt: args.prompt,
      script: args.script || undefined,
      schedule_type: args.schedule_type,
      schedule_value: args.schedule_value,
      context_mode: args.context_mode || 'group',
      targetJid,
      createdBy: groupFolder,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(TASKS_DIR, data);

    return {
      content: [
        {
          type: 'text' as const,
          text: `Task ${taskId} request queued: ${args.schedule_type} - ${args.schedule_value}`,
        },
      ],
    };
  },
);

server.tool(
  'list_tasks',
  "List all scheduled tasks. From main: shows all tasks. From other groups: shows only that group's tasks.",
  {},
  async () => {
    const tasksFile = path.join(IPC_DIR, 'current_tasks.json');

    try {
      if (!fs.existsSync(tasksFile)) {
        return {
          content: [
            {
              type: 'text' as const,
              text: 'Task snapshot unavailable; refresh it from the host.',
            },
          ],
          isError: true,
        };
      }

      const allTasks = readTaskSnapshot(tasksFile);

      const tasks = isMain
        ? allTasks
        : allTasks.filter(
            (t: { groupFolder: string }) => t.groupFolder === groupFolder,
          );

      if (tasks.length === 0) {
        return {
          content: [
            { type: 'text' as const, text: 'No scheduled tasks found.' },
          ],
        };
      }

      const formatted = tasks
        .map(
          (t) =>
            `- [${t.id}] ${t.prompt.slice(0, 50)}... (${t.schedule_type}: ${t.schedule_value}) - ${t.status}, next: ${t.next_run || 'N/A'}`,
        )
        .join('\n');

      return {
        content: [
          { type: 'text' as const, text: `Scheduled tasks:\n${formatted}` },
        ],
      };
    } catch (err) {
      if (!(err instanceof TaskSnapshotError) && !isFileSystemError(err))
        throw err;
      return {
        content: [
          {
            type: 'text' as const,
            text: 'Task snapshot unavailable; refresh the co-shipped host snapshot before use.',
          },
        ],
        isError: true,
      };
    }
  },
);

server.tool(
  'pause_task',
  'Pause a scheduled task. It will not run until resumed.',
  { task_id: z.string().describe('The task ID to pause') },
  async (args) => {
    const data = {
      type: 'pause_task',
      taskId: args.task_id,
      groupFolder,
      isMain,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(TASKS_DIR, data);

    return {
      content: [
        {
          type: 'text' as const,
          text: `Task ${args.task_id} pause requested.`,
        },
      ],
    };
  },
);

server.tool(
  'resume_task',
  'Resume a paused task.',
  { task_id: z.string().describe('The task ID to resume') },
  async (args) => {
    const data = {
      type: 'resume_task',
      taskId: args.task_id,
      groupFolder,
      isMain,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(TASKS_DIR, data);

    return {
      content: [
        {
          type: 'text' as const,
          text: `Task ${args.task_id} resume requested.`,
        },
      ],
    };
  },
);

server.tool(
  'cancel_task',
  'Cancel and delete a scheduled task.',
  { task_id: z.string().describe('The task ID to cancel') },
  async (args) => {
    const data = {
      type: 'cancel_task',
      taskId: args.task_id,
      groupFolder,
      isMain,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(TASKS_DIR, data);

    return {
      content: [
        {
          type: 'text' as const,
          text: `Task ${args.task_id} cancellation requested.`,
        },
      ],
    };
  },
);

server.tool(
  'update_task',
  'Update an existing scheduled task. Only provided fields are changed; omitted fields stay the same.',
  {
    task_id: z.string().describe('The task ID to update'),
    prompt: z.string().optional().describe('New prompt for the task'),
    schedule_type: z
      .enum(['cron', 'interval', 'once'])
      .optional()
      .describe('New schedule type'),
    schedule_value: z
      .string()
      .optional()
      .describe('New schedule value (see schedule_task for format)'),
    script: z
      .string()
      .optional()
      .describe(
        'New script for the task. Set to empty string to remove the script.',
      ),
  },
  async (args) => {
    if (args.schedule_type !== undefined || args.schedule_value !== undefined) {
      let type = args.schedule_type;
      let value = args.schedule_value;
      if (type === undefined || value === undefined) {
        try {
          const tasks = readTaskSnapshot(
            path.join(IPC_DIR, 'current_tasks.json'),
          );
          if (!Array.isArray(tasks))
            return {
              content: [
                {
                  type: 'text' as const,
                  text: 'Task schedule snapshot is unavailable; refresh tasks before updating.',
                },
              ],
              isError: true,
            };
          const existing = tasks.filter(
            (task: unknown) =>
              task !== null &&
              typeof task === 'object' &&
              'id' in task &&
              task.id === args.task_id,
          );
          if (existing.length !== 1)
            return {
              content: [
                {
                  type: 'text' as const,
                  text: 'Task schedule is unavailable or ambiguous; refresh tasks before updating.',
                },
              ],
              isError: true,
            };
          const task = existing[0];
          if (
            type === undefined &&
            (task.schedule_type === 'cron' ||
              task.schedule_type === 'once' ||
              task.schedule_type === 'interval')
          )
            type = task.schedule_type;
          if (value === undefined && typeof task.schedule_value === 'string')
            value = task.schedule_value;
        } catch (error) {
          if (
            !(error instanceof TaskSnapshotError) &&
            !isFileSystemError(error)
          )
            throw error;
          return {
            content: [
              {
                type: 'text' as const,
                text: 'Task schedule snapshot could not be read; refresh tasks before updating.',
              },
            ],
            isError: true,
          };
        }
      }
      const validationError = scheduleInputError(type ?? '', value ?? '');
      if (validationError)
        return {
          content: [{ type: 'text' as const, text: validationError }],
          isError: true,
        };
    }

    const data: Record<string, string | undefined> = {
      type: 'update_task',
      taskId: args.task_id,
      groupFolder,
      isMain: String(isMain),
      timestamp: new Date().toISOString(),
    };
    if (args.prompt !== undefined) data.prompt = args.prompt;
    if (args.script !== undefined) data.script = args.script;
    if (args.schedule_type !== undefined)
      data.schedule_type = args.schedule_type;
    if (args.schedule_value !== undefined)
      data.schedule_value = args.schedule_value;

    writeIpcFile(TASKS_DIR, data);

    return {
      content: [
        {
          type: 'text' as const,
          text: `Task ${args.task_id} update requested.`,
        },
      ],
    };
  },
);

server.tool(
  'register_group',
  `Register a new chat/group so the agent can respond to messages there. Main group only.

Use available_groups.json to find the JID for a group. The folder name must be channel-prefixed: "{channel}_{group-name}" (e.g., "whatsapp_family-chat", "telegram_dev-team", "discord_general"). Use lowercase with hyphens for the group name part.`,
  {
    jid: z
      .string()
      .describe(
        'The chat JID (e.g., "120363336345536173@g.us", "tg:-1001234567890", "dc:1234567890123456")',
      ),
    name: z.string().describe('Display name for the group'),
    folder: z
      .string()
      .describe(
        'Channel-prefixed folder name (e.g., "whatsapp_family-chat", "telegram_dev-team")',
      ),
    trigger: z.string().describe('Trigger word (e.g., "@Andy")'),
  },
  async (args) => {
    if (!isMain) {
      return {
        content: [
          {
            type: 'text' as const,
            text: 'Only the main group can register new groups.',
          },
        ],
        isError: true,
      };
    }

    const data = {
      type: 'register_group',
      jid: args.jid,
      name: args.name,
      folder: args.folder,
      trigger: args.trigger,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(TASKS_DIR, data);

    return {
      content: [
        {
          type: 'text' as const,
          text: `Group "${args.name}" registered. It will start receiving messages immediately.`,
        },
      ],
    };
  },
);

// Start the stdio transport
const transport = new StdioServerTransport();
await server.connect(transport);
