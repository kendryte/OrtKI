using Autofac;
using Autofac.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;

namespace OrtKITest;

public class Startup
{
    public IConfigurationRoot Configuration { get; set; }

    public void ConfigureHost(IHostBuilder hostBuilder) =>
        hostBuilder
            .ConfigureContainer<ContainerBuilder>(ConfigureContainer)
            .UseServiceProviderFactory(new AutofacServiceProviderFactory());
    
    private static void ConfigureContainer(ContainerBuilder builder)
    {
        builder.RegisterAssemblyModules(typeof(OrtKI.Tensor).Assembly);
    }
    
}