import { useState } from 'react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';

interface AppHeaderProps {
  activeTab?: string;
  onTabChange?: (tab: string) => void;
}

export const AppHeader = ({ activeTab = 'preview', onTabChange }: AppHeaderProps) => {
  return (
    <header className="h-14 border-b border-slate-200 bg-white flex items-center justify-between px-6 flex-shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-black flex items-center justify-center font-mono font-bold text-white text-sm">
            O
          </div>
          <span className="text-lg font-semibold font-mono text-slate-900">Ottic</span>
        </div>

        {/* Tabs Navigation */}
        <Tabs value={activeTab} onValueChange={onTabChange} className="w-auto">
          <TabsList className="bg-transparent h-14 p-0 gap-6">
            <TabsTrigger
              value="preview"
              className="bg-transparent data-[state=active]:bg-transparent data-[state=active]:shadow-none border-b-2 border-transparent data-[state=active]:border-black rounded-none px-0 pb-0.5 font-mono text-sm text-slate-600 data-[state=active]:text-slate-900 transition-colors"
            >
              Preview
            </TabsTrigger>
            <TabsTrigger
              value="artifacts"
              className="bg-transparent data-[state=active]:bg-transparent data-[state=active]:shadow-none border-b-2 border-transparent data-[state=active]:border-black rounded-none px-0 pb-0.5 font-mono text-sm text-slate-600 data-[state=active]:text-slate-900 transition-colors"
            >
              Artifacts
            </TabsTrigger>
            <TabsTrigger
              value="templates"
              className="bg-transparent data-[state=active]:bg-transparent data-[state=active]:shadow-none border-b-2 border-transparent data-[state=active]:border-black rounded-none px-0 pb-0.5 font-mono text-sm text-slate-600 data-[state=active]:text-slate-900 transition-colors"
            >
              Templates
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* User Menu */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" className="rounded-full h-8 w-auto px-3 gap-2 font-mono text-sm hover:bg-slate-100">
            <span className="text-slate-700">Perkup</span>
            <Avatar className="h-6 w-6 bg-black">
              <AvatarFallback className="bg-black text-white text-xs font-mono">
                P
              </AvatarFallback>
            </Avatar>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          <DropdownMenuLabel className="font-mono">My Account</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem className="font-mono">Profile</DropdownMenuItem>
          <DropdownMenuItem className="font-mono">Settings</DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem className="font-mono text-destructive">Log out</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </header>
  );
};
