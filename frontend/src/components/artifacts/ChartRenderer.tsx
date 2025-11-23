import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, RadialLinearScale, Title, Tooltip, Legend } from 'chart.js';
import { Bar, Line, Pie, Doughnut, Radar } from 'react-chartjs-2';
import type { ChartArtifactData } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend
);

interface ChartRendererProps {
  data: ChartArtifactData;
}

export const ChartRenderer = ({ data }: ChartRendererProps) => {
  const { chart_type, labels, datasets, title, x_axis_label, y_axis_label } = data;

  const chartData = {
    labels,
    datasets: datasets.map(ds => ({
      ...ds,
      backgroundColor: ds.backgroundColor || 'rgba(59, 130, 246, 0.5)',
      borderColor: ds.borderColor || 'rgba(59, 130, 246, 1)',
      borderWidth: ds.borderWidth || 1,
    })),
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        labels: {
          color: 'hsl(var(--foreground))',
        },
      },
    },
    scales: chart_type !== 'pie' && chart_type !== 'doughnut' ? {
      x: {
        title: {
          display: !!x_axis_label,
          text: x_axis_label,
          color: 'hsl(var(--foreground))',
        },
        ticks: {
          color: 'hsl(var(--muted-foreground))',
        },
        grid: {
          color: 'hsl(var(--border))',
        },
      },
      y: {
        title: {
          display: !!y_axis_label,
          text: y_axis_label,
          color: 'hsl(var(--foreground))',
        },
        ticks: {
          color: 'hsl(var(--muted-foreground))',
        },
        grid: {
          color: 'hsl(var(--border))',
        },
      },
    } : undefined,
  };

  const renderChart = () => {
    switch (chart_type) {
      case 'bar':
        return <Bar data={chartData} options={options} />;
      case 'line':
        return <Line data={chartData} options={options} />;
      case 'pie':
        return <Pie data={chartData} options={options} />;
      case 'doughnut':
        return <Doughnut data={chartData} options={options} />;
      case 'radar':
        return <Radar data={chartData} options={options} />;
      default:
        return <div>Unsupported chart type: {chart_type}</div>;
    }
  };

  return (
    <Card>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <div className="w-full" style={{ maxHeight: '400px' }}>
          {renderChart()}
        </div>
      </CardContent>
    </Card>
  );
};
