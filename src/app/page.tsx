import { DarkModeToggle } from "@/components/dark-mode-toggle";
import { Chat } from "@/components/chat";

export default function Home() {
  return (
    <main className="relative container flex min-h-screen flex-col">
      <div className="p-4 flex h-15 items-center justify-between supports-backdrop-blur:bg-background/60 sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur">
        <span className="font-bold">ICT Web Unit - Intranet Tools
        <br/><span className=""><small>Testing vector databases into conversational chat. Info @ Emanuele Strano</small></span>
        <br/><span className=""><small>This AI is trained on only 3 documents and the answers are limited to 3 vectors per query.</small></span>
        </span>
        <DarkModeToggle />
      </div>
      <div className="flex flex-1 py-4">
        <div className="w-full">
          <Chat />
        </div>
      </div>
    </main>
  );
}
