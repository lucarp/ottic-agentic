import { Card, CardContent } from '@/components/ui/card';

interface Template {
  id: string;
  title: string;
  description: string;
  icon: string;
  category: string;
}

const templates: Template[] = [
  {
    id: 'reddit-threads',
    title: 'Reddit Threads',
    description: 'Analyze Reddit discussions to discover trending topics and user sentiment in your niche',
    icon: '💬',
    category: 'Content Research',
  },
  {
    id: 'youtube-seo',
    title: 'YouTube SEO',
    description: 'Optimize your YouTube content with data-driven keywords and metadata recommendations',
    icon: '🎥',
    category: 'SEO',
  },
  {
    id: 'aeo-questions',
    title: 'AEO Questions',
    description: 'Generate Answer Engine Optimization questions to improve your content visibility',
    icon: '❓',
    category: 'Content Strategy',
  },
  {
    id: 'content-strategy',
    title: 'Content Strategy',
    description: 'Build comprehensive content strategies aligned with your business goals',
    icon: '📊',
    category: 'Strategy',
  },
  {
    id: 'product-discovery',
    title: 'Product Discovery',
    description: 'Research product opportunities and market gaps to guide your roadmap',
    icon: '🔍',
    category: 'Research',
  },
  {
    id: 'ads-intel',
    title: 'Ads Intel',
    description: 'Competitive advertising intelligence to inform your paid media strategy',
    icon: '🎯',
    category: 'Marketing',
  },
  {
    id: 'pillar-maps',
    title: 'Pillar Maps',
    description: 'Create topic clusters and content pillar structures for SEO dominance',
    icon: '🗺️',
    category: 'SEO',
  },
  {
    id: 'keyword-research',
    title: 'Keyword Research',
    description: 'Discover high-value keywords and search opportunities for your content',
    icon: '🔑',
    category: 'SEO',
  },
];

export const MainContent = () => {
  return (
    <div className="flex-1 overflow-auto bg-white">
      <div className="max-w-7xl mx-auto p-8">
        <div className="mb-8">
          <h2 className="text-2xl font-semibold font-mono mb-2 text-slate-900">Templates</h2>
          <p className="text-sm text-slate-600 font-mono">
            Start with a template to accelerate your marketing research and content creation
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {templates.map((template) => (
            <Card
              key={template.id}
              className="group cursor-pointer transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl border border-slate-200 relative overflow-hidden bg-gradient-to-br from-white to-slate-50"
            >
              {/* Green corner accent */}
              <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-[#06ff83]/15 via-transparent to-transparent rounded-bl-[100px]" />

              <CardContent className="p-6 relative">
                {/* Icon + Title */}
                <div className="flex items-start gap-3 mb-3">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center flex-shrink-0 border border-slate-300/50 group-hover:scale-110 group-hover:rotate-3 transition-all duration-300">
                    <span className="text-2xl">{template.icon}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold font-mono text-base leading-tight line-clamp-2 text-slate-900 group-hover:text-black transition-colors">
                      {template.title}
                    </h3>
                  </div>
                </div>

                {/* Description */}
                <p className="text-sm text-slate-600 leading-relaxed line-clamp-2 mb-3 min-h-[2.5rem]">
                  {template.description}
                </p>

                {/* Category Badge */}
                <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-slate-100 text-slate-700 border border-slate-300">
                  {template.category}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};
