export type ProjectCredentials = { projectId: string; recoveryKey: string }
export type Session = ProjectCredentials & { projectName: string }
export type Dataset = { id: string; name: string; status: string; image_count: number; schema_id: string }
