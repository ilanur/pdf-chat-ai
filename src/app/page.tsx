import { DarkModeToggle } from "@/components/dark-mode-toggle";
import { Chat } from "@/components/chat";

export default function Home() {
  return (
    <main className="relative container flex min-h-screen flex-col">
      <div className=" p-4 flex h-14 items-center justify-between supports-backdrop-blur:bg-background/60 sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur">
        <span className="font-bold">EUI Vectorized DBs into conversational chat<br/><span className=""><small>An AI Experiment by EUI Webunit - info at emanuele.strano@eui.eu</small></span></span>
        
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
