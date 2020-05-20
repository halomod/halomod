!! FORTRAN ROUTINES TO DO QUICK WORK ON LOOPS IN HALOMODEL
module hod_routines
contains

subroutine simps(dx, func,val)
    implicit none
    !!simps works on functions defined at EQUIDISTANT values of x only

    real(8), intent(in) :: dx      !The grid spacing
    real(8), intent(in) :: func(:) !The function
    real(8), intent(out):: val     !The value of the integral

    integer :: n !length of func
    integer :: i,end
    real(8) :: endbit

    n = size(func)

    if (mod(n,2)==0) then
        end =  n-1
        endbit = (func(n)+func(n-1))*dx/2
    else
        end=n
        endbit=0.d0
    end if

    val = sum(func(1:end-2:2) + func(3:end:2) + 4*func(2:end-1:2))
    val = val*dx/3 + endbit
end subroutine simps

subroutine trapz(dx,func,val)
    implicit none
    !!trapz works on functions defined at EQUIDISTANT values of x only

    real(8), intent(in) :: dx !The grid spacing
    real(8), intent(in) :: func(:) !The function
    real(8), intent(out) :: val !The value of the integral

    integer :: n !length of func
    integer :: i

    val = func(1)+func(n)
    do i=2,n-1
        val = val + 2*func(i)
    end do
    val = val*dx/2
end subroutine

!!!!!!!!!!!!! START ACTUAL CALLED ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine power_gal_2h(nlnk,nm,u,bias,nsat,ncen,dndm,mass,power)
    implicit none

    integer, intent(in) :: nlnk, nm !Size of lnk and mass arrays respectively
    real(8), intent(in) :: bias(nm), nsat(nm),ncen(nm),dndm(nm), mass(nm)
    real(8), intent(in) :: u(nlnk,nm)

    real(8), intent(out):: power(nlnk)

    real(8) ::mass_part(nm)
    integer :: i
    real(8) :: integrand(nm),dm

    integrand = 0.d0
    dm = log(mass(2))-log(mass(1))

    mass_part = ncen*dndm*bias*mass
    do i=1,nlnk
        integrand = (1.d0+u(i,:) *nsat)*mass_part !multiply by mass as we integrate in log-space
        call simps(dm,integrand,power(i))
      !  if (i==nlnk) then
      !      write(*,*) maxval(integrand),minval(integrand)
      !      write(*,*) maxval(mass_part/mass),minval(mass_part/mass)
      !      write(*,*) maxval(u(i,:)),minval(u(i,:))
      !  end if
    end do

end subroutine

subroutine power_gal_1h_ss(nlnk,nm,u,dndm,nsat,ncen,mass,central,power)
    implicit none

    integer, intent(in) :: nlnk, nm !Size of lnk and mass arrays respectively
    real(8), intent(in) :: ncen(nm),nsat(nm),dndm(nm), mass(nm)
    real(8), intent(in) :: u(nlnk,nm)
    logical, intent(in) :: central

    real(8), intent(out):: power(nlnk)

    integer :: i
    real(8) :: integrand(nm),dm
    integrand = 0.d0

    dm = log(mass(2))-log(mass(1))

    do i=1,nlnk
        if (central)then
            integrand = ncen*nsat**2 * u(i,:)**2*dndm*mass
        else
            integrand = nsat**2 * u(i,:)**2 *dndm*mass
        end if
        call simps(dm,integrand,power(i))
    end do
end subroutine

subroutine corr_gal_1h_ss(nr,nm,mass,dndm,ncen,nsat,lam,central,corr)
    implicit none

    integer, intent(in) :: nr,nm
    real(8), intent(in) :: mass(nm), ncen(nm),nsat(nm),dndm(nm)
    real(8), intent(in) :: lam(nr,nm)
    logical, intent(in) :: central

    real(8), intent(out) :: corr(nr)

    integer :: i
    real(8) :: integrand(nm),dm
    integrand = 0.d0
    dm = log(mass(2))-log(mass(1))

    do i=1,nr
        if (central)then
            integrand = dndm*ncen*nsat**2*lam(i,:)/mass !Not mass**2 since int in log space
        else
            integrand = dndm*nsat**2*lam(i,:)/mass
        end if

        call simps(dm,integrand,corr(i))
    end do
end subroutine

subroutine corr_gal_1h_cs(nr,nm,r,mass,dndm,ncen,nsat,rho,mean_dens,delta_halo,corr)
    implicit none

    integer, intent(in) :: nr,nm
    real(8), intent(in) :: r(nr), mass(nm), ncen(nm),nsat(nm),dndm(nm)
    real(8), intent(in) :: rho(nr,nm),mean_dens,delta_halo

    real(8), intent(out) :: corr(nr)

    integer :: i
    real(8) :: mmin
    real(8) :: integrand(nm),dm


    dm = log(mass(2))-log(mass(1))

    do i=1,nr
        integrand = 0.d0
        mmin = 4*3.14159265*r(i)**3*mean_dens*delta_halo/3
        where (mass>mmin) integrand = dndm*2*ncen*nsat*rho(i,:)*mass !*mass since in logspace
        call simps(dm,integrand,corr(i))
    end do
end subroutine

subroutine corr_gal_1h(nr,nm,r,mass,dndm,ncen,nsat,rho,lam,central,mean_dens,delta_halo,corr)
    implicit none

    integer, intent(in) :: nr,nm
    real(8), intent(in) :: r(nr), mass(nm), ncen(nm),nsat(nm),dndm(nm)
    real(8), intent(in) :: rho(nr,nm),mean_dens,delta_halo
    real(8), intent(in) :: lam(nr,nm)
    logical, intent(in) :: central

    real(8), intent(out) :: corr(nr)

    integer :: i,j
    real(8) :: mmin
    real(8) :: integrand(nm),dm

    dm = log(mass(2))-log(mass(1))

    !Integrating in log-space, so everything is multiplied by mass
    do i=1,nr
        integrand = 0.d0
        if (central)then
            integrand = dndm*ncen*nsat**2*lam(i,:)
        else
            integrand = dndm*nsat**2*lam(i,:)
        end if

        mmin = 4*3.14159265*r(i)**3*mean_dens*delta_halo/3

        where (mass>mmin) integrand = integrand+dndm*2*ncen*nsat*rho(i,:)

        integrand = integrand * mass
        call simps(dm,integrand,corr(i))
    end do
end subroutine

subroutine get_subfind_centres(nhalos,npart,groupoffsets,grouplen,pos,centres)
    implicit none
    !This is here because of a potentially very large loop.

    integer, intent(in) :: nhalos,npart
    integer, intent(in) :: groupoffsets(nhalos), grouplen(nhalos)
    real, intent(inout) :: pos(npart,3)

    real, intent(out) :: centres(nhalos,3)

    integer :: i,j,a,b
    real :: boxsize

    boxsize = ceiling(maxval(pos))

    do i=1,nhalos
        a = groupoffsets(i) + 1
        b = groupoffsets(i) + grouplen(i)

        ! First calculate a centre
        !centres(i,:) = sum(pos(a:b,:))/grouplen(i)

        do j=1,3
            where ((pos(a:b,j)-minval(pos(a:b,j)))>boxsize/2) pos(a:b,j) = pos(a:b,j) - boxsize
            centres(i,j) = sum(pos(a:b,j))/grouplen(i)
        end do


        if (i==nhalos) then
            write(*,*) centres(i,:)
        end if
        do j=1,3
            if (centres(i,j)<0.0)then
                centres(i,j) = centres(i,j) + boxsize
            end if
        end do
    end do


end subroutine
end module
